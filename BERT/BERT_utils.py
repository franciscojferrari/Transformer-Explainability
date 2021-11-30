import torch
from datasets import load_dataset
from torchmetrics import F1

class Rationale:
    ann_id: str
    docid: str
    start_token: int
    end_token: int

    def to_token_level(self) -> List['Rationale']:
        ret = []
        for t in range(self.start_token, self.end_token):
            ret.append(Rationale(self.ann_id, self.docid, t, t+1))
        return ret

    @classmethod
    def from_annotation(cls, ann: Annotation) -> List['Rationale']:
        ret = []
        for ev_group in ann.evidences:
            for ev in ev_group:
                ret.append(Rationale(ann.annotation_id, ev.docid, ev.start_token, ev.end_token))
        return ret

    @classmethod
    def from_instance(cls, inst: dict) -> List['Rationale']:
        ret = []
        for rat in inst['rationales']:
            for pred in rat.get('hard_rationale_predictions', []):
                ret.append(Rationale(inst['annotation_id'], rat['docid'], pred['start_token'], pred['end_token']))
        return ret

def _keyed_rationale_from_list(rats: List[Rationale]) -> Dict[Tuple[str, str], Rationale]:
    ret = defaultdict(set)
    for r in rats:
        ret[(r.ann_id, r.docid)].add(r)
    return ret

def partial_match_score(truth: List[Rationale], pred: List[Rationale], thresholds: List[float]) -> List[Dict[str, Any]]:
    """Computes a partial match F1
    Computes an instance-level (annotation) micro- and macro-averaged F1 score.
    True Positives are computed by using intersection-over-union and
    thresholding the resulting intersection-over-union fraction.
    Micro-average results are computed by ignoring instance level distinctions
    in the TP calculation (and recall, and precision, and finally the F1 of
    those numbers). Macro-average results are computed first by measuring
    instance (annotation + document) precisions and recalls, averaging those,
    and finally computing an F1 of the resulting average.
    """

    ann_to_rat = _keyed_rationale_from_list(truth)
    pred_to_rat = _keyed_rationale_from_list(pred)

    _f1 = F1()
    num_classifications = {k:len(v) for k,v in pred_to_rat.items()}
    num_truth = {k:len(v) for k,v in ann_to_rat.items()}
    ious = defaultdict(dict)
    for k in set(ann_to_rat.keys()) | set(pred_to_rat.keys()):
        for p in pred_to_rat.get(k, []):
            best_iou = 0.0
            for t in ann_to_rat.get(k, []):
                num = len(set(range(p.start_token, p.end_token)) & set(range(t.start_token, t.end_token)))
                denom = len(set(range(p.start_token, p.end_token)) | set(range(t.start_token, t.end_token)))
                iou = 0 if denom == 0 else num / denom
                if iou > best_iou:
                    best_iou = iou
            ious[k][p] = best_iou
    scores = []
    for threshold in thresholds:
        threshold_tps = dict()
        for k, vs in ious.items():
            threshold_tps[k] = sum(int(x >= threshold) for x in vs.values())
        micro_r = sum(threshold_tps.values()) / sum(num_truth.values()) if sum(num_truth.values()) > 0 else 0
        micro_p = sum(threshold_tps.values()) / sum(num_classifications.values()) if sum(num_classifications.values()) > 0 else 0
        micro_f1 = _f1(micro_r, micro_p)
        macro_rs = list(threshold_tps.get(k, 0.0) / n if n > 0 else 0 for k, n in num_truth.items())
        macro_ps = list(threshold_tps.get(k, 0.0) / n if n > 0 else 0 for k, n in num_classifications.items())
        macro_r = sum(macro_rs) / len(macro_rs) if len(macro_rs) > 0 else 0
        macro_p = sum(macro_ps) / len(macro_ps) if len(macro_ps) > 0 else 0
        macro_f1 = _f1(macro_r, macro_p)
        scores.append({'threshold': threshold,
                       'micro': {
                            'p': micro_p,
                            'r': micro_r,
                            'f1': micro_f1
                       },
                       'macro': {
                            'p': macro_p,
                            'r': macro_r,
                            'f1': macro_f1
                       },
                       })
    return scores

def score_hard_rationale_predictions(truth: List[Rationale], pred: List[Rationale]) -> Dict[str, Dict[str, float]]:
    """Computes instance (annotation)-level micro/macro averaged F1s"""
    _f1 = F1()
    scores = dict()
    truth = set(truth)
    pred = set(pred)
    micro_prec = len(truth & pred) / len(pred)
    micro_rec = len(truth & pred) / len(truth)
    micro_f1 = _f1(micro_prec, micro_rec)

    scores['instance_micro'] = {
                                'p': micro_prec,
                                'r': micro_rec,
                                'f1': micro_f1,
                               }

    ann_to_rat = _keyed_rationale_from_list(truth)
    pred_to_rat = _keyed_rationale_from_list(pred)
    instances_to_scores = dict()
    for k in set(ann_to_rat.keys()) | (pred_to_rat.keys()):
        if len(pred_to_rat.get(k, set())) > 0:
            instance_prec = len(ann_to_rat.get(k, set()) & pred_to_rat.get(k, set())) / len(pred_to_rat[k])
        else:
            instance_prec = 0
        if len(ann_to_rat.get(k, set())) > 0:
            instance_rec = len(ann_to_rat.get(k, set()) & pred_to_rat.get(k, set())) / len(ann_to_rat[k])
        else:
            instance_rec = 0
        instance_f1 = _f1(instance_prec, instance_rec)
        instances_to_scores[k] = {
                                    'p': instance_prec,
                                    'r': instance_rec,
                                    'f1': instance_f1,
                                 }
    # these are calculated as sklearn would
    macro_prec = sum(instance['p'] for instance in instances_to_scores.values()) / len(instances_to_scores)
    macro_rec = sum(instance['r'] for instance in instances_to_scores.values()) / len(instances_to_scores)
    macro_f1 = sum(instance['f1'] for instance in instances_to_scores.values()) / len(instances_to_scores)
    scores['instance_macro'] = {
                                'p': macro_prec,
                                'r': macro_rec,
                                'f1': macro_f1,
                               }
    return scores

def get_movie_rationales():
    train_dataset = load_dataset("movie_rationales", split="train")
    def get_hard_rationale(data, format=True, stagger=True):
        hard_rationales = []
        for i in range(len(data["review"])):
            hard_rationale = []
            for j in range(len(data["evidences"][i])):
                token_split = data["evidences"][i][j].split()
                for k in range(len(token_split)):
                    index_in_review = data["review"][i].find(token_split[k])
                    if index_in_review not in hard_rationale:
                        if stagger:
                            hard_rationale.append(index_in_review + i * 512)
                        else:
                            hard_rationale.append(index_in_review)
            hard_rationales.append(hard_rationale)
        return {"hard_rationale": hard_rationales}
    train_dataset = train_dataset.map(get_hard_rationale, batched=True)
    if format:
        train_dataset.set_format("torch")
    return train_dataset["hard_rationale"]

if __name__ == "__main__":
    pass