"""Compare pre-merge business records without emitting personal data."""
import hashlib,json
from pathlib import Path
from verify_console_merge import connect,ROOT,OUT

before=json.loads((Path.home()/'.local/share/xunbian-backend/backups/before-console-merge-20261102.json').read_text())
result={'tables':{},'changedOriginalRows':0}
with connect() as db,db.cursor() as c:
    for table,oldrows in before.items():
        c.execute('SHOW KEYS FROM `'+table+"` WHERE Key_name='PRIMARY'")
        primary=[r['Column_name'] for r in sorted(c.fetchall(),key=lambda r:r['Seq_in_index'])]
        c.execute('SELECT * FROM `'+table+'`')
        current={tuple(row[k] for k in primary):json.loads(json.dumps(row,default=str)) for row in c.fetchall()}
        changed=0
        for row in oldrows:
            actual=current.get(tuple(row[k] for k in primary))
            if actual is None or any(actual.get(k)!=v for k,v in row.items()):changed+=1
        result['tables'][table]={'originalRows':len(oldrows),'changedOriginalRows':changed}
        result['changedOriginalRows']+=changed
    c.execute('SELECT COUNT(*) AS n FROM reference_assets');result['archiveAssetIndexes']=c.fetchone()['n']
    c.execute("SELECT COUNT(*) AS n FROM sample_tasks t JOIN batches b ON b.id=t.batch_id WHERE b.source_type='ARCHIVE'");result['fabricatedArchiveTasks']=c.fetchone()['n']
    c.execute("SELECT COUNT(*) AS n FROM prediction_results p JOIN batches b ON b.id=p.batch_id WHERE b.source_type='ARCHIVE'");result['fabricatedArchivePredictions']=c.fetchone()['n']
    c.execute('SELECT console_json FROM reference_sources LIMIT 1');archive=json.loads(c.fetchone()['console_json']);result['archiveSamples']=sum(len(r['items']) for r in archive['regions'])
result['sourceFiles']={name:hashlib.sha256((ROOT/'web'/name).read_bytes()).hexdigest() for name in ('index.html','app_data.json')}
result['passed']=result['changedOriginalRows']==0 and result['archiveSamples']==5984 and result['archiveAssetIndexes']==23936 and result['fabricatedArchiveTasks']==result['fabricatedArchivePredictions']==0
(OUT/'preservation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
assert result['passed'], 'Original data changed or historical provenance invalid; see preservation.json'
