"""Original conversation and analysis services over one authorized batch snapshot.

The caller owns batch authorization, snapshot/version namespaces and trusted
``_authenticatedUser`` / ``sessionKey`` binding. No client history or generated
SQL is promoted into server state by this adapter.
"""
from pathlib import Path

from analysis_tasks import AnalysisTasks
from conversation_assistant import ConversationAssistant
from model_sql import ModelSQL


class BatchAssistant:
    def __init__(self, source_path, history_dir, *, generate, model_id,
                 query_update_adapter=False):
        history = Path(history_dir)
        self.sql = ModelSQL(
            source_path, history / 'model_sql.sqlite', generate=generate,
            model_id=model_id, contract_mode=True,
            query_update_adapter=query_update_adapter)
        if self.sql.task_type == 'building_segmentation':
            self.sql.query_update_adapter = False
        self._conversation = ConversationAssistant(
            self.sql, history / 'assistant_conversations.sqlite')
        self._analysis = AnalysisTasks(
            self.sql.source_path, history / 'analysis_tasks.sqlite',
            generate=generate, model_id=model_id,
            reviewed_provider=self.sql.reviewed_provider)

    def run(self, payload):
        result=self._conversation.run(payload)
        if self.sql.task_type=='building_segmentation' and result.get('usedSQL') and 'estimated_building_count' in (result.get('sqlQuery') or ''):
            result['countingNotice']='建筑量为面积≥10像素的8连通区域估计数；粘连与边界截断可能影响计数，跨图累加未作地理去重。'
            # A single verified scalar can be stated directly, without asking a
            # second model to invent arithmetic or infer building instances.
            import sqlglot
            from sqlglot import exp
            tree=sqlglot.parse_one(result['sqlQuery'],read='sqlite')
            rows=result.get('sqlRows') or []
            if isinstance(tree,exp.Select) and len(tree.expressions)==1 and len(rows)==1 and len(rows[0])==1:
                projection=tree.expressions[0]
                if isinstance(projection,exp.Alias):projection=projection.this
                if isinstance(projection,exp.Sum):projection=projection.this
                value=next(iter(rows[0].values()))
                if isinstance(projection,exp.Column) and projection.name=='estimated_building_count' and isinstance(value,(int,float)):
                    result['answer']=f'当前查询范围的建筑量为{value:g}个（候选区域估计）。'
        return result

    def analysis(self, payload):
        return self._analysis.handle(payload)
