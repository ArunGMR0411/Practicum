#!/usr/bin/env python3
"""Multimodal detection reviewer for wizard runs.

Opens a browser UI over a run's detections/ folder.
Colours: face=green, screen=blue, text=amber.
Saves edits back to detections.jsonl + face/screen/text CSVs.
"""

from __future__ import annotations

import json
import socket
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from privacy_pipeline_app.wizard_workflow import (
    load_detection_records,
    refresh_state_from_detections,
    write_detection_artifacts,
)

# Active servers: run_dir -> (server, thread, port)
_SERVERS: dict[str, tuple[ThreadingHTTPServer, threading.Thread, int]] = {}

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Detection Review</title>
<style>
  :root {
    --face: #16a34a; --screen: #2563eb; --text: #d97706;
    --bg: #0f172a; --panel: #1e293b; --ink: #e2e8f0; --muted: #94a3b8; --line: #334155;
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui,sans-serif; background:var(--bg); color:var(--ink); height:100vh; overflow:hidden; }
  .layout { display:grid; grid-template-columns: 280px 1fr; height:100vh; }
  aside { background:var(--panel); border-right:1px solid var(--line); padding:12px; display:flex; flex-direction:column; gap:10px; overflow:hidden; }
  main { padding:12px; display:flex; flex-direction:column; gap:8px; overflow:hidden; }
  h1 { font-size:16px; margin:0; }
  .legend span { display:inline-flex; align-items:center; gap:6px; margin-right:10px; font-size:12px; color:var(--muted); }
  .sw { width:12px; height:12px; border-radius:3px; display:inline-block; }
  .btns { display:flex; flex-wrap:wrap; gap:6px; }
  button { border:1px solid var(--line); background:#0f172a; color:var(--ink); border-radius:8px; padding:7px 10px; cursor:pointer; font-size:12px; }
  button.primary { background:#2563eb; border-color:#2563eb; }
  button.ok { background:#15803d; border-color:#15803d; }
  button.active { outline:2px solid #fff; }
  button:disabled { opacity:.4; cursor:not-allowed; }
  #list { flex:1; overflow:auto; border:1px solid var(--line); border-radius:8px; }
  .item { padding:8px; border-bottom:1px solid var(--line); cursor:pointer; font-size:12px; }
  .item.active { background:#334155; }
  .item small { color:var(--muted); display:block; }
  #status { font-size:12px; color:var(--muted); min-height:1.2em; }
  #wrap { flex:1; overflow:hidden; display:flex; align-items:center; justify-content:center; background:#020617; border-radius:10px; border:1px solid var(--line); }
  canvas { max-width:100%; max-height:100%; cursor:crosshair; }
  .mode { font-size:11px; color:var(--muted); }
</style>
</head>
<body>
<div class="layout">
  <aside>
    <h1>Review detections</h1>
    <div class="legend">
      <span><i class="sw" style="background:var(--face)"></i>Face</span>
      <span><i class="sw" style="background:var(--screen)"></i>Screen</span>
      <span><i class="sw" style="background:var(--text)"></i>Text</span>
    </div>
    <div class="btns">
      <button id="prev">←</button>
      <button id="next">→</button>
      <button id="save" class="primary" title="Write CSVs + JSONL">Save</button>
      <button id="done" class="ok" title="Save and mark review complete">Done</button>
    </div>
    <div class="btns">
      <button id="mFace" class="active" data-m="face">Face</button>
      <button id="mScreen" data-m="screen">Screen</button>
      <button id="mText" data-m="text">Text</button>
    </div>
    <div class="btns">
      <button id="add" title="Drag on image to add box">+ Box</button>
      <button id="del" title="Delete selected box">Delete</button>
    </div>
    <div class="mode" id="modeHint">Select a box · drag to move · corners to resize</div>
    <div id="status">Loading…</div>
    <div id="list"></div>
  </aside>
  <main>
    <div id="meta" class="mode"></div>
    <div id="wrap"><canvas id="cv"></canvas></div>
  </main>
</div>
<script>
const COLORS = {face:'#16a34a', screen:'#2563eb', text:'#d97706'};
let tasks = [];
let idx = 0;
let modality = 'face';
let selected = -1;
let img = new Image();
let adding = false;
let drag = null; // {kind:'move'|'resize'|'new', corner?, startX, startY, ox1,oy1,ox2,oy2}
const canvas = document.getElementById('cv');
const ctx = canvas.getContext('2d');
let scale = 1;

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function boxesOf(task) {
  if (modality === 'face') return task.faces;
  if (modality === 'screen') return task.screens;
  return task.texts;
}

function setStatus(t){ document.getElementById('status').textContent = t; }

function renderList() {
  const el = document.getElementById('list');
  el.innerHTML = '';
  tasks.forEach((t,i) => {
    const d = document.createElement('div');
    d.className = 'item' + (i===idx?' active':'');
    d.innerHTML = `<strong>${t.image_id}</strong><small>F${t.faces.length} · S${t.screens.length} · T${t.texts.length}</small>`;
    d.onclick = () => { idx=i; selected=-1; loadImage(); renderList(); };
    el.appendChild(d);
  });
}

function loadImage() {
  const t = tasks[idx];
  if (!t) return;
  document.getElementById('meta').textContent = t.image_id;
  img.onload = () => { draw(); };
  img.src = '/image?id=' + encodeURIComponent(t.image_id);
}

function draw() {
  const t = tasks[idx];
  if (!t || !img.naturalWidth) return;
  const maxW = document.getElementById('wrap').clientWidth - 8;
  const maxH = document.getElementById('wrap').clientHeight - 8;
  scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1);
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  for (const [mod, arr] of [['face',t.faces],['screen',t.screens],['text',t.texts]]) {
    const col = COLORS[mod];
    arr.forEach((b,i) => {
      const x1=b.x1*scale, y1=b.y1*scale, x2=b.x2*scale, y2=b.y2*scale;
      ctx.lineWidth = (mod===modality && i===selected) ? 3 : 2;
      ctx.strokeStyle = col;
      ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      ctx.fillStyle = col + '33';
      ctx.fillRect(x1,y1,x2-x1,y2-y1);
      if (mod===modality && i===selected) {
        const hs=6;
        [[x1,y1],[x2,y1],[x1,y2],[x2,y2]].forEach(([x,y])=>{
          ctx.fillStyle='#fff'; ctx.fillRect(x-hs/2,y-hs/2,hs,hs);
        });
      }
    });
  }
}

function toImg(x,y){ return [x/scale, y/scale]; }

function hitBox(mx,my) {
  const arr = boxesOf(tasks[idx]);
  for (let i=arr.length-1;i>=0;i--) {
    const b=arr[i];
    if (mx>=b.x1 && mx<=b.x2 && my>=b.y1 && my<=b.y2) return i;
  }
  return -1;
}

function hitCorner(mx,my,b) {
  const tol = 10/scale;
  const pts = [[b.x1,b.y1,'nw'],[b.x2,b.y1,'ne'],[b.x1,b.y2,'sw'],[b.x2,b.y2,'se']];
  for (const [x,y,c] of pts) if (Math.abs(mx-x)<=tol && Math.abs(my-y)<=tol) return c;
  return null;
}

canvas.addEventListener('mousedown', e => {
  const t = tasks[idx]; if (!t) return;
  const rect = canvas.getBoundingClientRect();
  const [mx,my] = toImg(e.clientX-rect.left, e.clientY-rect.top);
  const arr = boxesOf(t);
  if (adding) {
    drag = {kind:'new', startX:mx, startY:my};
    return;
  }
  if (selected>=0 && selected<arr.length) {
    const c = hitCorner(mx,my,arr[selected]);
    if (c) { drag={kind:'resize', corner:c, ox1:arr[selected].x1, oy1:arr[selected].y1, ox2:arr[selected].x2, oy2:arr[selected].y2}; return; }
  }
  const hit = hitBox(mx,my);
  if (hit>=0) {
    selected = hit;
    const b = arr[hit];
    drag = {kind:'move', startX:mx, startY:my, ox1:b.x1, oy1:b.y1, ox2:b.x2, oy2:b.y2};
    draw();
  } else {
    selected = -1; draw();
  }
});

canvas.addEventListener('mousemove', e => {
  if (!drag) return;
  const t = tasks[idx];
  const arr = boxesOf(t);
  const rect = canvas.getBoundingClientRect();
  const [mx,my] = toImg(e.clientX-rect.left, e.clientY-rect.top);
  if (drag.kind==='new') {
    draw();
    ctx.strokeStyle = COLORS[modality];
    ctx.strokeRect(drag.startX*scale, drag.startY*scale, (mx-drag.startX)*scale, (my-drag.startY)*scale);
    return;
  }
  const b = arr[selected];
  if (!b) return;
  if (drag.kind==='move') {
    const dx=mx-drag.startX, dy=my-drag.startY;
    b.x1=Math.round(drag.ox1+dx); b.y1=Math.round(drag.oy1+dy);
    b.x2=Math.round(drag.ox2+dx); b.y2=Math.round(drag.oy2+dy);
  } else if (drag.kind==='resize') {
    let {ox1,oy1,ox2,oy2}=drag;
    if (drag.corner.includes('n')) oy1=my;
    if (drag.corner.includes('s')) oy2=my;
    if (drag.corner.includes('w')) ox1=mx;
    if (drag.corner.includes('e')) ox2=mx;
    b.x1=Math.round(Math.min(ox1,ox2)); b.x2=Math.round(Math.max(ox1,ox2));
    b.y1=Math.round(Math.min(oy1,oy2)); b.y2=Math.round(Math.max(oy1,oy2));
  }
  draw();
});

canvas.addEventListener('mouseup', e => {
  if (!drag) return;
  const t = tasks[idx];
  const arr = boxesOf(t);
  if (drag.kind==='new') {
    const rect = canvas.getBoundingClientRect();
    const [mx,my] = toImg(e.clientX-rect.left, e.clientY-rect.top);
    const x1=Math.round(Math.min(drag.startX,mx)), x2=Math.round(Math.max(drag.startX,mx));
    const y1=Math.round(Math.min(drag.startY,my)), y2=Math.round(Math.max(drag.startY,my));
    if (x2-x1>4 && y2-y1>4) {
      arr.push({x1,y1,x2,y2,score:1.0});
      selected = arr.length-1;
    }
    adding=false;
    document.getElementById('add').classList.remove('active');
  }
  // clamp counts
  t.face_count = t.faces.length;
  t.text_count = t.texts.length;
  t.screen_count = t.screens.length;
  drag=null;
  draw();
  renderList();
});

document.getElementById('add').onclick = () => {
  adding = !adding;
  document.getElementById('add').classList.toggle('active', adding);
  setStatus(adding ? 'Drag on image to draw a new box' : 'Ready');
};
document.getElementById('del').onclick = () => {
  const t = tasks[idx];
  const arr = boxesOf(t);
  if (selected>=0 && selected<arr.length) {
    arr.splice(selected,1);
    selected=-1;
    t.face_count=t.faces.length; t.text_count=t.texts.length; t.screen_count=t.screens.length;
    draw(); renderList();
  }
};
document.getElementById('prev').onclick = () => { if(idx>0){idx--; selected=-1; loadImage(); renderList();} };
document.getElementById('next').onclick = () => { if(idx<tasks.length-1){idx++; selected=-1; loadImage(); renderList();} };
['mFace','mScreen','mText'].forEach(id => {
  document.getElementById(id).onclick = (e) => {
    modality = e.target.dataset.m;
    ['mFace','mScreen','mText'].forEach(x => document.getElementById(x).classList.remove('active'));
    e.target.classList.add('active');
    selected=-1; draw();
  };
});
document.getElementById('save').onclick = async () => {
  await api('/save', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({tasks})});
  setStatus('Saved to detections/');
};
document.getElementById('done').onclick = async () => {
  await api('/save', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({tasks, mark_done:true})});
  setStatus('Saved · review marked done · you can close this tab');
};

async function boot() {
  const data = await api('/tasks');
  tasks = data.tasks;
  renderList();
  loadImage();
  setStatus(`${tasks.length} images`);
}
window.addEventListener('resize', draw);
boot();
</script>
</body>
</html>
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_detection_reviewer(run_dir: str, host: str = "127.0.0.1") -> str:
    """Start (or reuse) a reviewer server for this run. Returns browser URL."""
    run_path = Path(run_dir).resolve()
    key = str(run_path)
    if key in _SERVERS:
        _srv, _th, port = _SERVERS[key]
        return f"http://{host}:{port}/"

    records = load_detection_records(run_path)
    if not records:
        raise FileNotFoundError(f"No detections found in {run_path / 'detections'}")

    state: dict[str, Any] = {"records": records, "run_dir": run_path}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send(HTTPStatus.OK, HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/tasks":
                payload = json.dumps({"tasks": state["records"]}).encode("utf-8")
                self._send(HTTPStatus.OK, payload, "application/json")
                return
            if parsed.path == "/image":
                qs = parse_qs(parsed.query)
                image_id = (qs.get("id") or [""])[0]
                rec = next((r for r in state["records"] if r["image_id"] == image_id), None)
                if not rec:
                    self._send(HTTPStatus.NOT_FOUND, b"missing", "text/plain")
                    return
                path = Path(rec["local_path"])
                if not path.exists():
                    self._send(HTTPStatus.NOT_FOUND, b"file missing", "text/plain")
                    return
                data = path.read_bytes()
                ctype = "image/webp" if path.suffix.lower() == ".webp" else "image/jpeg"
                if path.suffix.lower() == ".png":
                    ctype = "image/png"
                self._send(HTTPStatus.OK, data, ctype)
                return
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/save":
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            tasks = payload.get("tasks") or []
            # normalise counts
            for t in tasks:
                t["faces"] = t.get("faces") or []
                t["texts"] = t.get("texts") or []
                t["screens"] = t.get("screens") or []
                t["face_count"] = len(t["faces"])
                t["text_count"] = len(t["texts"])
                t["screen_count"] = len(t["screens"])
            state["records"] = tasks
            write_detection_artifacts(run_path / "detections", tasks)
            refresh_state_from_detections(str(run_path))
            if payload.get("mark_done"):
                from privacy_pipeline_app.wizard_workflow import mark_review_done

                mark_review_done(str(run_path))
            self._send(HTTPStatus.OK, json.dumps({"ok": True}).encode("utf-8"), "application/json")

    port = _free_port()
    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _SERVERS[key] = (server, thread, port)
    return f"http://{host}:{port}/"
