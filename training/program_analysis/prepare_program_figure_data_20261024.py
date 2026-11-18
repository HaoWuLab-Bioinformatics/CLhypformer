"""Read existing counts and frozen program usage; export patient-equal gene gradients."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

MEDIUM=Path(__file__).resolve().parent
RUNS=MEDIUM.parent/'bio_runs'
PROGRAM=RUNS/'gse176078_patient_programs_20261022'
CONFIRM=RUNS/'gse176078_program_confirm_20261023'
OUT=RUNS/'gse176078_program_figures_20261024/source_data'


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    panels=json.loads((PROGRAM/'B_panels.json').read_text())
    genes_requested=list(dict.fromkeys(g for p in panels for g in p['genes']))
    ambiguous=set(json.loads((CONFIRM/'identity_audit.json').read_text())['ambiguous_ids'])
    rows=[];audits=[]
    cases=['source_test','SeuratObject_ERTotal','SeuratObject_HER2','SeuratObject_TNBC']
    for case in cases:
        print('PREPARE',case,flush=True)
        if case=='source_test':
            base=RUNS/'gse176078_full_20261017'
            with np.load(PROGRAM/'source_program_usage.npz') as z:
                keep=z['role']==2;idx=z['idx'][keep];usage=z['usage'][keep]
            with np.load(base/'data/fold_01/biomed/gse176078_brca.npz') as z:
                patients=z['patient_ids'][idx].astype(str)
            genes=np.load(base/'data/gse176078_brca_genes.npy').astype(str)
            counts=sp.load_npz(base/'data/gse176078_brca_counts.npz').tocsr()[idx]
            cohort='GSE176078';eligible=np.ones(len(idx),bool)
        else:
            base=RUNS/'gse176078_caf_mechanism_20261017/external/source'/case
            with np.load(CONFIRM/f'external_targets/{case}.npz') as z:
                idx=z['idx'].copy();usage=z['usage'].copy();patients=z['patients'].astype(str)
            genes=np.load(base/'genes.npy').astype(str)
            counts=sp.load_npz(base/'counts.npz').tocsr()[idx]
            cohort='GSE161529';eligible=~np.isin(patients,list(ambiguous))
        lookup={g:i for i,g in enumerate(genes)}
        assert all(g in lookup for g in genes_requested)
        totals=np.asarray(counts.sum(1)).ravel()
        expression=np.log1p(counts[:,[lookup[g] for g in genes_requested]].toarray()*
                           (1e4/np.maximum(totals,1))[:,None])
        before=len(patients);retained=0;retained_patients=0
        for patient in np.unique(patients):
            mask=(patients==patient)&eligible
            n=int(mask.sum())
            if n<30:continue
            retained+=n;retained_patients+=1
            local=expression[mask]
            mean=local.mean(0);sd=local.std(0)
            z=(local-mean)/np.maximum(sd,1e-6)
            for panel in panels:
                program=panel['program'];activity=usage[mask,program]
                # Equal-activity ties stay together; do not invent an ordering.
                bins=np.minimum(((rankdata(activity,method='average')-.5)/n*5).astype(int),4)
                for quintile in range(5):
                    selected=bins==quintile
                    for gene in panel['genes']:
                        j=genes_requested.index(gene)
                        rows.append(dict(cohort=cohort,object=case,patient=patient,program=program,
                            gene=gene,quintile=quintile+1,cells=int(selected.sum()),patient_cells=n,
                            expression_variable=bool(sd[j]>1e-6),
                            mean_within_patient_z=float(z[selected,j].mean()) if selected.any() else None))
        audits.append(dict(cohort=cohort,object=case,all_malignant_cells=before,
            plotted_cells=retained,plotted_patients=retained_patients,
            exclusion='Predefined identity ambiguity and <30 malignant cells per patient',
            all_B_genes_retained=True,cell_sampling=False))
        del counts,expression
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'patient_gene_quintiles.csv',index=False)
    average=frame.groupby(['cohort','program','gene','quintile']).agg(
        mean_z=('mean_within_patient_z','mean'),
        contributing_patients=('mean_within_patient_z','count'),total_cells=('cells','sum')).reset_index()
    average.to_csv(OUT/'gene_gradient_means.csv',index=False)
    (OUT/'gene_gradient_integrity.json').write_text(json.dumps(dict(cases=audits,
        axis='Within-patient quintiles of A-derived frozen NMF program usage, not neural-model embeddings',
        gene_values='LogCP10k, gene-wise within-patient z scores; then patient-equal bin means',
        missing_bins='Retained as missing, never forced into nonempty bins',
        constant_gene_policy='Retained as zero z-score and flagged',
        new_model_training=False,new_program_discovery=False),indent=2))
    print(json.dumps(audits),flush=True)


if __name__=='__main__':run()
