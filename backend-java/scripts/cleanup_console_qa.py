"""Disable only three exact temporary identities from the private merge fixture."""
import json
from verify_console_merge import PRIVATE,OUT,connect,client,call
people=json.loads(PRIVATE.read_text());admin=client(people['admin']);members={k:client(people[k]) for k in ('owner','editor')}
for role in ('owner','editor'):call(admin,'POST','/admin/users/'+people[role]['id']+'/enabled',json={'enabled':False})
revoked=all(s.get('http://127.0.0.1:8786/api/v1/auth/me',timeout=10).status_code==401 for s in members.values())
assert revoked
with connect() as db,db.cursor() as c:
    db.begin()
    c.execute('SELECT id FROM app_bootstrap WHERE id=1 FOR UPDATE')
    c.execute('SELECT username,role FROM app_users WHERE id=%s FOR UPDATE',(people['admin']['id'],));row=c.fetchone()
    assert row['username']==people['admin']['username'] and row['username'].startswith('qa_merge_admin_') and row['role']=='ADMIN'
    c.execute("SELECT COUNT(*) AS n FROM app_users WHERE role='ADMIN' AND enabled=TRUE AND id<>%s",(people['admin']['id'],));assert c.fetchone()['n']>=1
    c.execute('UPDATE app_users SET enabled=FALSE,auth_version=auth_version+1 WHERE id=%s',(people['admin']['id'],));db.commit()
assert admin.get('http://127.0.0.1:8786/api/v1/auth/me',timeout=10).status_code==401
result={'passed':True,'temporaryAccountsDisabled':3,'existingSessionsRevoked':True,'originalAdministratorUnchanged':True}
(OUT/'qa-cleanup.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
