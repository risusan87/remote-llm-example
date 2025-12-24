
import hashlib
from typing import List, Tuple, Set, Dict

from django.apps import apps

try:
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    import torch
    SEARCH_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    tokenizer = AutoTokenizer.from_pretrained(SEARCH_MODEL)
    model = AutoModel.from_pretrained(SEARCH_MODEL)
except ImportError:
    pass


# @inproceedings{reimers-2019-sentence-bert,
#     title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
#     author = "Reimers, Nils and Gurevych, Iryna",
#     booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
#     month = "11",
#     year = "2019",
#     publisher = "Association for Computational Linguistics",
#     url = "http://arxiv.org/abs/1908.10084",
# }
# Licensed under Apache License 2.0
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2


def _vectorize(text: str) -> List[float]:
    tokenized_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        model_output = model(**tokenized_input)
    # mean pooling
    token_embeddings = model_output[0]
    input_mask_expanded = tokenized_input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a).reshape(-1)
    b = np.array(b).reshape(-1)
    print(a.shape, b.shape)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# this may be able to move under app config
def get_searchable_models() -> Dict[str, List[str]]:
    context_map = None
    for Model in apps.get_models():
        # is this model searchable? i.e. is model decorated with @searchable_content?
        if not hasattr(Model, '_searchable'):
            continue 
        if context_map is None:
            context_map = {}
        context_fields = [attribute_ns for attribute_ns, _ in Model._searchable]
        for f in context_fields:
            model_ns = f.split('.')[0]
            # Model = apps.get_model('backend', model_ns)
            if context_map.get(model_ns) is None:
                context_map[model_ns] = []
            context_map[model_ns].append(f.split('.')[1])
    return context_map

def catch_up(refresh=False):
    """
    pokes Document model and its relations up to date.
    calling this manually is redundant but harmless.

    execution guarantees all entry in database that are marked as searchable context vectorized and stored + binding.

    :param refresh: completely repopulate stakeholders if set to True. this should be False for most of the time.
    """
    # search engine
    record_ct = 0
    record_dup = 0

    # this section is particularly for vector search
    Document = apps.get_model('backend', 'Document')
    ContentDocument = apps.get_model('backend', 'ContentDocument')
    if refresh:
        Document.objects.all().delete()
        # ContentDocument is also deleted by cascade

    for model_ns, field_ns_list in get_searchable_models().items():
        Model = apps.get_model('backend', model_ns)
        for entry in Model.objects.only(*field_ns_list):
            pk = entry.pk
            # print(f'Primary keys are expected to be int -> type is: {type(pk)}')
            for field_ns in field_ns_list:
                context = getattr(entry, field_ns)
                # at this point, model is searchable, context is a context field
                # lets check if all context is documented (vectorized and stored)

                # here, check if the context is already in the Document model
                # no vectorization yet to reduce the cost
                hash_comp = hashlib.sha256(context.encode()).hexdigest()
                record = Document.objects.filter(hash=hash_comp).first()
                if not record:
                    # if not, we need to vectorize and store it
                    record = Document(
                        hash=hash_comp,
                        vector=_vectorize(context).tolist(),
                    )
                    record.save()
                    # bind
                    relation_record = ContentDocument(
                        target_model_ns=model_ns,
                        entry_pk=pk,
                        document=record,
                    )
                    relation_record.save()
                    record_ct += 1
                else:
                    record_dup += 1

    return record_ct, record_dup


def searchable_content(contexts: List[Tuple[str, float]]):
    """
    Decorates a model class to be searchable.
    """
    def decorator(cls):
        cls._searchable = contexts
        # save() override
        save = cls.save
        def or_save(self, *args, **kwargs):
            result = save(self, *args, **kwargs)
            catch_up()
            return result
        cls.save = or_save
        return cls
    return decorator

def vec_search(query: str, search_category: str='Any', threshold: float=0.5, top_k=10) -> Set[str]:
    """
    Search for the given query
    Uses cosine similarity
    """
    # cleanup if necessary
    query_vector = _vectorize(query)
    Document = apps.get_model('backend', 'Document')
    ContentDocument = apps.get_model('backend', 'ContentDocument')
    documents = Document.objects.all()
    scores = {}
    for context in documents:
        score = _cosine_similarity(query_vector, context.vector)
        if score > threshold:
            scores[context] = score
    match = []
    for score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        content = ContentDocument.objects.filter(document=score[0]).first()
        Model = apps.get_model('backend', content.target_model_ns)
        entry = Model.objects.filter(pk=content.entry_pk).first()
        if not entry:
            continue
        match.append(entry.__str__())
    return set(match[:top_k])
        


