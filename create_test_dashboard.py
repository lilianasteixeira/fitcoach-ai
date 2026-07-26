#!/usr/bin/env python3
import json
import urllib.request
import base64

dash = {
  "title": "FitCoach Test",
  "panels": [
    {
      "type": "stat",
      "title": "Total conversations",
      "id": 1,
      "gridPos": {"x": 0, "y": 0, "w": 8, "h": 8},
      "targets": [{
        "refId": "A",
        "datasource": {"type": "grafana-postgresql-datasource", "uid": "PE7AC3E30A56FEE85"},
        "rawSql": "SELECT count(*) AS value FROM conversations"
      }]
    },
    {
      "type": "table",
      "title": "All conversations",
      "id": 2,
      "gridPos": {"x": 8, "y": 0, "w": 16, "h": 12},
      "targets": [{
        "refId": "A",
        "datasource": {"type": "grafana-postgresql-datasource", "uid": "PE7AC3E30A56FEE85"},
        "rawSql": "SELECT id, created_at, question, feedback FROM conversations ORDER BY created_at DESC"
      }]
    }
  ],
  "refresh": "10s",
  "time": {"from": "now-30d", "to": "now"},
  "timezone": "browser"
}

payload = {"dashboard": dash, "folderId": 0, "overwrite": True}
data = json.dumps(payload).encode()
req = urllib.request.Request('http://localhost:3000/api/dashboards/db', data=data, headers={'Content-Type':'application/json'})
req.add_header('Authorization', 'Basic ' + base64.b64encode(b'admin:admin').decode())

try:
    with urllib.request.urlopen(req) as r:
        res = json.loads(r.read())
        print(f"SUCCESS: {res['status']} - {res['url']}")
except urllib.error.HTTPError as e:
    print(f"ERROR {e.code}: {e.read().decode()[:500]}")
