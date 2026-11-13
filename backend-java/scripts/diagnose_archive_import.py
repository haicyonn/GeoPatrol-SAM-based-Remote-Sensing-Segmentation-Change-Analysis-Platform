from verify_console_merge import connect,ROOT
import json
with connect() as db,db.cursor() as c:
    c.execute('SELECT @@max_allowed_packet AS max_packet,@@net_read_timeout AS read_timeout,@@net_write_timeout AS write_timeout')
    print(c.fetchone())
    c.execute("SELECT COUNT(*) AS archives FROM reference_sources")
    print(c.fetchone())
print('source bytes', (ROOT/'web/app_data.json').stat().st_size)
