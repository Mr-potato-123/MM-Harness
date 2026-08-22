"""Dependency-free toy monitor for local solve runs.

It deliberately has no start/plan/orchestration controls. It only displays the
backend's status/probe stream and submits operator suggestions or an interrupt
command to the local human-control store.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from m2harness.human_control import HumanControlStore


_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>M2Harness Toy Monitor</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#111827;color:#e5e7eb;margin:0}
header{padding:18px 24px;background:#1f2937;border-bottom:1px solid #374151;position:sticky;top:0;z-index:2}
h1{font-size:20px;margin:0 0 4px}small{color:#9ca3af}
main{padding:20px 24px;display:grid;gap:16px;max-width:1180px;margin:auto}
.card{background:#1f2937;border:1px solid #374151;border-radius:10px;padding:16px;box-shadow:0 4px 16px #0003}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.grow{flex:1}
.badge{padding:3px 8px;border-radius:999px;background:#374151;font-size:12px}.running{background:#075985}.blocked,.failed{background:#991b1b}.completed{background:#166534}
button{border:0;border-radius:6px;padding:8px 12px;cursor:pointer;color:white;background:#2563eb}button.danger{background:#dc2626}button:disabled{opacity:.45;cursor:not-allowed}
textarea{width:100%;min-height:70px;box-sizing:border-box;background:#111827;color:#e5e7eb;border:1px solid #4b5563;border-radius:6px;padding:8px;margin:10px 0}
pre{white-space:pre-wrap;word-break:break-word;background:#111827;border-radius:6px;padding:10px;max-height:260px;overflow:auto;font-size:12px;color:#cbd5e1}
.muted{color:#9ca3af}.empty{padding:30px;text-align:center;color:#9ca3af}
</style></head>
<body><header><h1>M2Harness Toy Monitor</h1><small>只监视后端 solve 进程，并提交建议/中断；不负责启动或编排任务。</small></header>
<main id="app"><div class="empty">正在读取运行状态…</div></main>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(url,opt){let r=await fetch(url,opt);return await r.json()}
async function refresh(){
 const runs=await api('/api/runs'); const app=document.querySelector('#app');
 if(!runs.length){app.innerHTML='<div class="empty">暂无后端 run。请先在项目终端启动 solve 工作流。</div>';return}
 app.innerHTML='';
 for(const x of runs){const ev=await api('/api/runs/'+encodeURIComponent(x.run_id)+'/events');
  const status=esc(x.status||'unknown'), stage=esc(x.event||'-');
  const card=document.createElement('section');card.className='card';
  card.innerHTML=`<div class="row"><strong>${esc(x.task_id||'-')}</strong><span class="badge ${status}">${status}</span><span class="badge">${stage}</span><span class="muted grow">run ${esc(x.run_id)}</span></div>
  <p class="muted">actor=${esc(x.actor||'-')} · iteration=${esc(x.iteration||'-')} · updated=${esc(x.updated_at||'-')}</p>
  <textarea placeholder="给当前 solve 工具的建议（将在下一个安全阶段注入 Model/Code 上下文）"></textarea>
  <div class="row"><button class="suggest">提交建议</button><button class="danger interrupt">请求中断</button></div>
  <details><summary>最近探针事件</summary><pre>${esc(ev.slice(-12).map(e=>JSON.stringify(e)).join('\n'))}</pre></details>`;
  card.querySelector('.suggest').onclick=async()=>{const t=card.querySelector('textarea');if(t.value.trim()){await api('/api/runs/'+encodeURIComponent(x.run_id)+'/suggest',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({message:t.value,task_id:x.task_id})});t.value='';await refresh()}};
  card.querySelector('.interrupt').onclick=async()=>{if(confirm('请求停止当前 solve？')){await api('/api/runs/'+encodeURIComponent(x.run_id)+'/interrupt',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({reason:'operator requested interrupt',task_id:x.task_id})});await refresh()}};
  app.appendChild(card);
 }
}
refresh();setInterval(refresh,2000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    store: HumanControlStore

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path == "/":
            self._send(HTTPStatus.OK, _HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/runs":
            self._json(self.store.list_status())
            return
        prefix = "/api/runs/"
        if parsed.path.startswith(prefix) and parsed.path.endswith("/events"):
            run_id = unquote(parsed.path[len(prefix):-len("/events")]).strip("/")
            self._json(self.store.probe_events(run_id))
            return
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        prefix = "/api/runs/"
        if not parsed.path.startswith(prefix):
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        suffix = parsed.path[len(prefix):]
        if suffix.endswith("/suggest"):
            run_id = unquote(suffix[:-len("/suggest")]).strip("/")
            payload = self._body()
            command = self.store.submit_suggestion(run_id, str(payload.get("message", "")), task_id=payload.get("task_id"))
            self._json(command.__dict__, HTTPStatus.ACCEPTED)
            return
        if suffix.endswith("/interrupt"):
            run_id = unquote(suffix[:-len("/interrupt")]).strip("/")
            payload = self._body()
            command = self.store.request_interrupt(run_id, str(payload.get("reason", "operator requested interrupt")), task_id=payload.get("task_id"))
            self._json(command.__dict__, HTTPStatus.ACCEPTED)
            return
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _body(self) -> dict:
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 16_384)
            value = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            value = {}
        return value if isinstance(value, dict) else {}

    def _json(self, value, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False), "application/json; charset=utf-8")

    def _send(self, status: HTTPStatus, body: str, content_type: str) -> None:
        raw = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, _format: str, *_args) -> None:
        return


def serve_toy_ui(workspace_root: Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    store = HumanControlStore(workspace_root)
    handler = type("ToyMonitorHandler", (_Handler,), {"store": store})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"M2Harness Toy Monitor: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
