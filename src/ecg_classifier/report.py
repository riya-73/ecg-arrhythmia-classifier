"""Render the recorded DS2-only measurements as a Markdown report."""
import json
from pathlib import Path

def main():
    m=json.loads(Path('results/metrics.json').read_text())
    def row(name,v):
        n=v['classification_report']['Normal']; a=v['classification_report']['Abnormal']
        return f"| {name} | {v['accuracy']:.4f} | {v['roc_auc']:.4f} | {v['pr_auc']:.4f} | {n['precision']:.4f} / {n['recall']:.4f} / {n['f1-score']:.4f} | {a['precision']:.4f} / {a['recall']:.4f} / {a['f1-score']:.4f} |"
    txt=['# MIT-BIH DS2 held-out results','',f"Test beats: {m['metadata']['counts']['test']:,} (Normal {m['metadata']['test_class_counts']['normal']:,}; Abnormal {m['metadata']['test_class_counts']['abnormal']:,}).","","| Model | Accuracy | ROC-AUC | PR-AUC | Normal P / R / F1 | Abnormal P / R / F1 |","|---|---:|---:|---:|---:|---:|",row('1D CNN',m['cnn']),row('Random forest',m['random_forest']),'','Confusion matrices are in `figures/confusion_matrices.png`. Metrics use DS2 only; model selection/early stopping uses DS1 validation.','']
    Path('results/results.md').write_text('\n'.join(txt))
if __name__=='__main__': main()
