"""Publish only this explicit, reviewed list of final game-result artifacts."""
import hashlib
import json
from pathlib import Path
import wandb


def read(name):
    return json.loads(Path(name).read_text(encoding='utf-8-sig'))


def main():
    public=read('results/final_public.json')
    holdout=read('results/final_holdout.json')
    validation=read('results/validation_final.json')
    selected=read('results/selected_model.json')
    checkpoint=Path('models/model_astro_party.zip')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest==public['sha256']==holdout['sha256']==validation['sha256']
    # No folders, credentials, environment dumps or old experiment streams.
    report_files=[Path('../output/pdf/Astro_Party_Report.pdf'),Path('REPORT_TW.md'),
        Path('results/figures/training_curves.png'),Path('results/figures/final_results.png'),
        Path('results/final_public.json'),Path('results/final_holdout.json'),
        Path('results/validation_final.json')]
    assert all(path.is_file() for path in report_files)
    api=wandb.Api(timeout=30)
    entity=api.default_entity
    run=wandb.init(project='astro-party-hw3',entity=entity,name='final-public95-91-holdout92-87',
                   job_type='final-report',mode='online',dir='.',config=selected)
    url=run.url
    try:
        metrics={}
        for split,data in [('public',public),('holdout',holdout)]:
            for bot,result in data['results'].items():
                metrics[f'{split}/{bot}_win_rate']=result['win_rate']
        metrics.update({f'validation/{bot}_win_rate':validation['metrics'][f'eval/{bot}_win_rate']
                        for bot in ['rusher','sniper']})
        run.log(metrics)
        run.summary.update(dict(model_sha256=digest,parameter_count=validation['parameters'],
            exported_global_step=selected['selected_step'],public_score=10,
            source_training_run=f'{entity}/astro-party-hw3/6o6gk5el'))
        run.log({'figures/training':wandb.Image('results/figures/training_curves.png'),
                 'figures/final_results':wandb.Image('results/figures/final_results.png')})
        table=wandb.Table(columns=['split','opponent','seed','win','steps'])
        for split,data in [('public',public),('holdout',holdout)]:
            for bot,result in data['results'].items():
                for row in result['rows']:
                    table.add_data(split,bot,row['seed'],row['win'],row['steps'])
        run.log({'evaluation_games':table})
        model_artifact=wandb.Artifact('AstroParty-final-model',type='model',
                                     metadata=dict(sha256=digest,global_step=selected['selected_step']))
        model_artifact.add_file(str(checkpoint),name='model_astro_party.zip')
        model_artifact.add_file('results/selected_model.json',name='selected_model.json')
        run.log_artifact(model_artifact)
        report_artifact=wandb.Artifact('AstroParty-final-report',type='report')
        for path in report_files:
            report_artifact.add_file(str(path),name=path.name)
        run.log_artifact(report_artifact)
    finally:
        run.finish()
    remote=wandb.Api(timeout=30).run(f'{entity}/astro-party-hw3/{run.id}')
    artifacts=[a.name for a in remote.logged_artifacts()]
    assert remote.summary['public/sniper_win_rate']==.91
    assert len(artifacts)>=2
    result=dict(published=True,url=url,run_id=run.id,artifacts=artifacts,model_sha256=digest)
    Path('results/wandb_final.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
