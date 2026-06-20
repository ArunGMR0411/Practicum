"""Browser reviewer for face-box and text-presence review workflows."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

def resolve_project_path(value: str | Path) -> Path:
    """Resolve relative project paths while preserving absolute paths."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


FACE_PACK_ROOT = resolve_project_path(
    os.environ.get(
        "CASTLE_FACE_PACK_ROOT",
        str(PROJECT_ROOT / "data" / "castle2024" / "annotations" / "face_detection" / "02_egocentric_stress_500"),
    )
)
FACE_TASKS_CSV_PATH = resolve_project_path(
    os.environ.get(
        "CASTLE_FACE_TASKS_CSV",
        str(PROJECT_ROOT / "data" / "thesis_manifests" / "final_face_detection_500.csv"),
    )
)
FACE_PROPOSALS_CSV_PATH = FACE_PACK_ROOT / "manifest.csv"
FACE_OUTPUT_CSV_PATH = FACE_PACK_ROOT / "manifest.csv"
FACE_REVIEW_STATUS_CSV_PATH = FACE_PACK_ROOT / "manifest.csv"
FACE_CATEGORY_CSV_PATH = FACE_PACK_ROOT / "manifest.csv"

DEFAULT_MULTIMODAL_TASKS_CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "01_protocol"
    / "annotations"
    / "multimodal_250"
    / "reviewed_multimodal_250_with_boxes.csv"
)


def resolve_text_review_paths() -> tuple[Path, Path, Path]:
    """Resolve the shared multimodal-review CSV, optionally from the environment."""
    configured_path = (
        os.environ.get("CASTLE_MULTIMODAL_REVIEW_CSV", "").strip()
        or os.environ.get("CASTLE_TEXT_REVIEW_CSV", "").strip()
    )
    csv_path = Path(configured_path) if configured_path else DEFAULT_MULTIMODAL_TASKS_CSV_PATH
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    pack_root = csv_path.parent
    return pack_root, csv_path, csv_path


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CASTLE Reviewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg: #f3efe6;
      --panel: #fffaf0;
      --ink: #162028;
      --accent: #9b3d12;
      --accent-soft: #efc6b2;
      --line: #d8c8b4;
      --ok: #2d6a4f;
      --warn: #8d6708;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f7f1e8 0%, #efe4d2 100%);
    }
    .shell {
      display: grid;
      grid-template-columns: 360px 1fr;
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: rgba(255, 250, 240, 0.94);
      padding: 16px;
      overflow: auto;
    }
    .workspace {
      padding: 18px;
      overflow: auto;
    }
    h1, h2, h3 { margin: 0 0 10px; }
    h1 { font-size: 24px; color: var(--accent); }
    .meta, .status, .review-panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 14px;
    }
    .controls, .box-controls, .binary-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }
    button, select, input, textarea {
      font: inherit;
    }
    button {
      border: 1px solid var(--accent);
      background: #fff;
      color: var(--accent);
      padding: 8px 12px;
      border-radius: 999px;
      cursor: pointer;
      font-size: 14px;
    }
    button.primary {
      background: var(--accent);
      color: white;
    }
    button.secondary-ok {
      border-color: var(--ok);
      color: var(--ok);
    }
    button.layer-active {
      color: white;
      background: var(--ink);
      border-color: var(--ink);
    }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .task-list {
      max-height: 40vh;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
    }
    .task-row {
      padding: 10px 12px;
      border-bottom: 1px solid #eee3d7;
      cursor: pointer;
    }
    .task-row.active {
      background: var(--accent-soft);
    }
    .task-row small { display: block; color: #5d666c; }
    #canvasWrap {
      position: relative;
      display: inline-block;
      border: 1px solid var(--line);
      background: white;
      border-radius: 12px;
      padding: 10px;
      margin-bottom: 14px;
    }
    canvas {
      display: block;
      max-width: 100%;
      cursor: crosshair;
    }
    .legend {
      margin-top: 10px;
      color: #56616a;
      font-size: 14px;
    }
    .box-list {
      margin-top: 12px;
      border: 1px solid var(--line);
      background: white;
      border-radius: 10px;
      overflow: auto;
      max-height: 24vh;
    }
    .box-row {
      display: grid;
      grid-template-columns: 28px 1fr auto;
      gap: 10px;
      padding: 8px 10px;
      border-bottom: 1px solid #eee3d7;
      align-items: center;
    }
    .box-row.active { background: #fff0e7; }
    .pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      background: #edf6f1;
      color: var(--ok);
      font-size: 12px;
      margin-left: 8px;
    }
    .pill.pending {
      background: #fbf1d3;
      color: var(--warn);
    }
    .review-grid {
      display: grid;
      grid-template-columns: 170px 1fr;
      gap: 10px;
      align-items: center;
    }
    textarea {
      width: 100%;
      min-height: 92px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: white;
    }
    select, input[type="text"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: white;
    }
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <h1 id="appTitle">CASTLE Reviewer</h1>
      <div class="meta" id="summary"></div>
      <div class="controls">
        <button id="prevBtn">Prev</button>
        <button id="nextBtn">Next</button>
        <button id="saveBtn" class="primary">Save CSV</button>
      </div>
      <div class="controls">
        <button id="markReviewedBtn" class="secondary-ok">Mark Reviewed</button>
        <button id="markAllReviewedBtn" class="secondary-ok">Mark All Reviewed</button>
        <button id="addBoxBtn">Add Box</button>
        <button id="deleteBoxBtn">Delete Box</button>
      </div>
      <div class="controls hidden" id="multimodalLayerControls">
        <button id="textLayerBtn" type="button">Text layer</button>
        <button id="screenLayerBtn" type="button">Screen layer</button>
      </div>
      <div class="status" id="status">Loading…</div>
      <div class="task-list" id="taskList"></div>
    </aside>
    <main class="workspace">
      <div class="meta" id="imageMeta"></div>
      <div id="canvasWrap">
        <canvas id="canvas"></canvas>
      </div>
      <div class="legend" id="legendBox">
        Click a box in the list or on the image to select it. Drag inside a box to move it. Drag corners to resize. Use Add Box, then drag on the image to create a new proposal.
      </div>
      <div class="review-panel hidden" id="faceCategoryPanel">
        <h3>Face and Scene Category Review</h3>
        <div class="review-grid">
          <label for="faceCountCategory">Face count category</label>
          <select id="faceCountCategory">
            <option value="">Choose…</option>
            <option value="no_face">no_face</option>
            <option value="single_face">single_face</option>
            <option value="multi_face">multi_face</option>
            <option value="uncertain">uncertain</option>
          </select>
          <label for="faceScaleCategory">Face scale category</label>
          <select id="faceScaleCategory">
            <option value="">Choose…</option>
            <option value="none">none</option>
            <option value="very_small_or_distant">very_small_or_distant</option>
            <option value="small">small</option>
            <option value="medium">medium</option>
            <option value="large">large</option>
            <option value="mixed_scale">mixed_scale</option>
            <option value="uncertain">uncertain</option>
          </select>
          <label for="edgePartialFace">Edge or partial face</label>
          <select id="edgePartialFace">
            <option value="">Choose…</option>
            <option value="yes">yes</option>
            <option value="no">no</option>
            <option value="uncertain">uncertain</option>
          </select>
          <label for="profileOccludedFace">Profile or occluded face</label>
          <select id="profileOccludedFace">
            <option value="">Choose…</option>
            <option value="yes">yes</option>
            <option value="no">no</option>
            <option value="uncertain">uncertain</option>
          </select>
          <label for="downwardEgocentricView">Downward egocentric view</label>
          <select id="downwardEgocentricView">
            <option value="">Choose…</option>
            <option value="yes">yes</option>
            <option value="no">no</option>
            <option value="uncertain">uncertain</option>
          </select>
          <label for="blurLowSharpness">Motion blur / low sharpness</label>
          <select id="blurLowSharpness">
            <option value="">Choose…</option>
            <option value="yes">yes</option>
            <option value="no">no</option>
            <option value="uncertain">uncertain</option>
          </select>
          <label for="lowLightDim">Low light / dim</label>
          <select id="lowLightDim">
            <option value="">Choose…</option>
            <option value="yes">yes</option>
            <option value="no">no</option>
            <option value="uncertain">uncertain</option>
          </select>
          <label for="clutterLevel">Clutter level</label>
          <select id="clutterLevel">
            <option value="">Choose…</option>
            <option value="low">low</option>
            <option value="medium">medium</option>
            <option value="high">high</option>
            <option value="uncertain">uncertain</option>
          </select>
          <label for="outdoorVehicleScene">Outdoor / vehicle scene</label>
          <select id="outdoorVehicleScene">
            <option value="">Choose…</option>
            <option value="yes">yes</option>
            <option value="no">no</option>
            <option value="uncertain">uncertain</option>
          </select>
          <label for="categoryReviewStatus">Category review status</label>
          <select id="categoryReviewStatus">
            <option value="pending">pending</option>
            <option value="reviewed">reviewed</option>
          </select>
          <label for="categoryNotes">Category notes</label>
          <textarea id="categoryNotes" placeholder="Optional notes for category corrections"></textarea>
        </div>
      </div>
      <div class="review-panel hidden" id="textReviewPanel">
        <h3>Text Review</h3>
        <div class="review-grid">
          <label for="manualTextPresent">Visible text present</label>
          <select id="manualTextPresent">
            <option value="">Choose…</option>
            <option value="yes">yes</option>
            <option value="no">no</option>
          </select>
          <label for="manualLegibleText">Contains legible text</label>
          <select id="manualLegibleText">
            <option value="">Choose…</option>
            <option value="yes">yes</option>
            <option value="no">no</option>
          </select>
          <label for="manualScreenPresent">Visible screen present</label>
          <select id="manualScreenPresent">
            <option value="">Choose…</option>
            <option value="yes">yes</option>
            <option value="no">no</option>
          </select>
          <label for="manualScreenContent">Readable or potentially sensitive screen content</label>
          <select id="manualScreenContent">
            <option value="">Choose…</option>
            <option value="yes">yes</option>
            <option value="no">no</option>
          </select>
          <label for="reviewStatus">Review status</label>
          <select id="reviewStatus">
            <option value="pending">pending</option>
            <option value="reviewed">reviewed</option>
          </select>
          <label for="reviewerId">Reviewer ID</label>
          <input id="reviewerId" type="text" placeholder="e.g. arun" />
          <label for="manualNotes">Notes</label>
          <textarea id="manualNotes" placeholder="Optional notes for ambiguous frames"></textarea>
        </div>
      </div>
      <div class="box-list" id="boxList"></div>
    </main>
  </div>
<script>
const state = {
  mode: null,
  tasks: [],
  index: 0,
  image: new Image(),
  scale: 1,
  selectedBox: null,
  interactionMode: 'idle',
  dragStart: null,
  activeHandle: null,
  activeBoxType: 'text',
};

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function isFaceMode() {
  return state.mode === 'face_boxes';
}

function isTextMode() {
  return state.mode === 'text_presence';
}

function isScreenMode() {
  return state.mode === 'screen_presence';
}

function isMultimodalMode() {
  return state.mode === 'multimodal_presence';
}

function isBinaryReviewMode() {
  return isTextMode() || isScreenMode() || isMultimodalMode();
}

function canEditBoxes() {
  return isFaceMode() || isTextMode() || isScreenMode() || isMultimodalMode();
}

function activeBoxes() {
  const task = currentTask();
  if (isMultimodalMode()) {
    return state.activeBoxType === 'text' ? task.text_boxes : task.screen_boxes;
  }
  return task.boxes || [];
}

function setActiveBoxType(boxType) {
  state.activeBoxType = boxType;
  state.selectedBox = null;
  state.interactionMode = 'idle';
  document.getElementById('textLayerBtn').classList.toggle('layer-active', boxType === 'text');
  document.getElementById('screenLayerBtn').classList.toggle('layer-active', boxType === 'screen');
  document.getElementById('addBoxBtn').textContent = boxType === 'text' ? 'Add Text Box' : 'Add Screen Box';
  render();
}

function setStatus(message) {
  document.getElementById('status').textContent = message;
}

function currentTask() {
  return state.tasks[state.index];
}

function updateSummary() {
  const reviewed = state.tasks.filter(t => t.reviewed).length;
  if (isMultimodalMode()) {
    const textBoxes = state.tasks.reduce((sum, t) => sum + t.text_boxes.length, 0);
    const screenBoxes = state.tasks.reduce((sum, t) => sum + t.screen_boxes.length, 0);
    document.getElementById('summary').innerHTML = `
      <strong>${state.tasks.length}</strong> images<br>
      <strong>${reviewed}</strong> reviewed<br>
      <strong style="color:#0066cc">${textBoxes}</strong> text boxes<br>
      <strong style="color:#228b22">${screenBoxes}</strong> screen boxes
    `;
    return;
  }
  if (isFaceMode() || isTextMode() || isScreenMode()) {
    const boxCount = state.tasks.reduce((sum, t) => sum + (t.boxes ? t.boxes.length : 0), 0);
    const type = isTextMode() ? 'text' : (isScreenMode() ? 'screen' : 'face');
    document.getElementById('summary').innerHTML = `
      <strong>${state.tasks.length}</strong> images<br>
      <strong>${reviewed}</strong> reviewed<br>
      <strong>${boxCount}</strong> ${type} boxes total
    `;
  } else {
    const yesCount = isTextMode()
      ? state.tasks.filter(t => t.manual_text_present === 'yes').length
      : state.tasks.filter(t => t.manual_screen_present === 'yes').length;
    document.getElementById('summary').innerHTML = `
      <strong>${state.tasks.length}</strong> images<br>
      <strong>${reviewed}</strong> reviewed<br>
      <strong>${yesCount}</strong> marked present
    `;
  }
}

function taskRowDetail(task) {
  if (isFaceMode()) {
    const faceCategory = task.face_count_category || 'unset';
    const scaleCategory = task.face_scale_category || 'unset';
    return `${task.condition_label} · ${task.boxes.length} boxes · faces=${faceCategory} · scale=${scaleCategory}`;
  }
  if (isMultimodalMode()) {
    return `text=${task.text_boxes.length} · screen=${task.screen_boxes.length} · text visible=${task.manual_text_present || 'unset'} · screen visible=${task.manual_screen_present || 'unset'}`;
  }
  const manual = isTextMode()
    ? (task.manual_text_present || 'unset')
    : (task.manual_screen_present || 'unset');
  const proxy = isTextMode() ? task.visible_text_flag : task.visible_screen_flag;
  return `${task.sample_bucket} · proxy=${proxy ? 'yes' : 'no'} · manual=${manual}`;
}

function renderTaskList() {
  const list = document.getElementById('taskList');
  list.innerHTML = '';
  state.tasks.forEach((task, idx) => {
    const row = document.createElement('div');
    row.className = 'task-row' + (idx === state.index ? ' active' : '');
    const pillClass = task.reviewed ? 'pill' : 'pill pending';
    const pillText = task.reviewed ? 'reviewed' : 'pending';
    row.innerHTML = `
      <div>${task.relative_path}<span class="${pillClass}">${pillText}</span></div>
      <small>${taskRowDetail(task)}</small>
    `;
    row.onclick = () => loadTask(idx);
    list.appendChild(row);
  });
}

function renderBoxList() {
  const list = document.getElementById('boxList');
  if (!canEditBoxes()) {
    list.innerHTML = '';
    list.classList.add('hidden');
    return;
  }
  list.classList.remove('hidden');
  list.innerHTML = '';
  const layers = isMultimodalMode()
    ? [['text', currentTask().text_boxes], ['screen', currentTask().screen_boxes]]
    : [[isTextMode() ? 'text' : (isScreenMode() ? 'screen' : 'face'), activeBoxes()]];
  layers.forEach(([typeLabel, boxes]) => boxes.forEach((box, idx) => {
    const row = document.createElement('div');
    const selected = idx === state.selectedBox && (!isMultimodalMode() || typeLabel === state.activeBoxType);
    row.className = 'box-row' + (selected ? ' active' : '');
    const typeColor = typeLabel === 'text' ? '#0066cc' : (typeLabel === 'screen' ? '#228b22' : '#9b3d12');
    row.innerHTML = `
      <strong style="color:${typeColor}">${idx + 1} (${typeLabel})</strong>
      <div>x1=${box.x1}, y1=${box.y1}, x2=${box.x2}, y2=${box.y2}<br><small>score=${box.score ?? ''}</small></div>
      <button>Pick</button>
    `;
    row.querySelector('button').onclick = () => {
      if (isMultimodalMode()) setActiveBoxType(typeLabel);
      state.selectedBox = idx;
      render();
    };
    list.appendChild(row);
  }));
}

function renderMeta() {
  const task = currentTask();
  if (isFaceMode()) {
    document.getElementById('imageMeta').innerHTML = `
      <strong>${task.relative_path}</strong><br>
      ${task.camera_stream_id} · ${task.day_id} · ${task.view_type} · ${task.condition_label}<br>
      faces=${task.face_count_category || 'unset'} · scale=${task.face_scale_category || 'unset'}
    `;
  } else if (isMultimodalMode()) {
    document.getElementById('imageMeta').innerHTML = `
      <strong>${task.relative_path}</strong><br>
      ${task.camera_stream_id} · ${task.day_id} · ${task.view_type}<br>
      <span style="color:#0066cc">text boxes=${task.text_boxes.length}</span> ·
      <span style="color:#228b22">screen boxes=${task.screen_boxes.length}</span>
    `;
  } else {
    const proxy = isTextMode() ? task.visible_text_flag : task.visible_screen_flag;
    const proxyLabel = isTextMode() ? 'visible_text' : 'visible_screen';
    document.getElementById('imageMeta').innerHTML = `
      <strong>${task.relative_path}</strong><br>
      ${task.camera_stream_id} · ${task.day_id} · ${task.view_type} · ${task.condition_label}<br>
      proxy ${proxyLabel}=${proxy ? 'yes' : 'no'} · bucket=${task.sample_bucket}
    `;
  }
}

function syncTextFormFromTask() {
  if (!isBinaryReviewMode()) return;
  const task = currentTask();
  document.getElementById('manualTextPresent').value = isScreenMode()
    ? (task.manual_screen_present || '')
    : (task.manual_text_present || '');
  document.getElementById('manualLegibleText').value = isScreenMode()
    ? (task.manual_screen_contains_sensitive_content || '')
    : (task.manual_contains_legible_text || '');
  document.getElementById('manualScreenPresent').value = task.manual_screen_present || '';
  document.getElementById('manualScreenContent').value = task.manual_screen_contains_sensitive_content || '';
  document.getElementById('reviewStatus').value = task.review_status || 'pending';
  document.getElementById('reviewerId').value = task.reviewer_id || '';
  document.getElementById('manualNotes').value = task.manual_notes || '';
}

function syncFaceCategoryFormFromTask() {
  if (!isFaceMode()) return;
  const task = currentTask();
  document.getElementById('faceCountCategory').value = task.face_count_category || '';
  document.getElementById('faceScaleCategory').value = task.face_scale_category || '';
  document.getElementById('edgePartialFace').value = task.edge_partial_face || '';
  document.getElementById('profileOccludedFace').value = task.profile_occluded_face || '';
  document.getElementById('downwardEgocentricView').value = task.downward_egocentric_view || '';
  document.getElementById('blurLowSharpness').value = task.blur_low_sharpness || '';
  document.getElementById('lowLightDim').value = task.low_light_dim || '';
  document.getElementById('clutterLevel').value = task.clutter_level || '';
  document.getElementById('outdoorVehicleScene').value = task.outdoor_vehicle_scene || '';
  document.getElementById('categoryReviewStatus').value = task.category_review_status || 'pending';
  document.getElementById('categoryNotes').value = task.category_notes || '';
}

function syncTaskFromFaceCategoryForm() {
  if (!isFaceMode()) return;
  const task = currentTask();
  task.face_count_category = document.getElementById('faceCountCategory').value;
  task.face_scale_category = document.getElementById('faceScaleCategory').value;
  task.edge_partial_face = document.getElementById('edgePartialFace').value;
  task.profile_occluded_face = document.getElementById('profileOccludedFace').value;
  task.downward_egocentric_view = document.getElementById('downwardEgocentricView').value;
  task.blur_low_sharpness = document.getElementById('blurLowSharpness').value;
  task.low_light_dim = document.getElementById('lowLightDim').value;
  task.clutter_level = document.getElementById('clutterLevel').value;
  task.outdoor_vehicle_scene = document.getElementById('outdoorVehicleScene').value;
  task.category_review_status = document.getElementById('categoryReviewStatus').value;
  task.category_notes = document.getElementById('categoryNotes').value;
  updateSummary();
  renderTaskList();
}

function syncTaskFromTextForm() {
  if (!isBinaryReviewMode()) return;
  const task = currentTask();
  if (isTextMode() || isMultimodalMode()) {
    task.manual_text_present = document.getElementById('manualTextPresent').value;
    task.manual_contains_legible_text = document.getElementById('manualLegibleText').value;
  }
  if (isScreenMode()) {
    task.manual_screen_present = document.getElementById('manualTextPresent').value;
    task.manual_screen_contains_sensitive_content = document.getElementById('manualLegibleText').value;
  } else if (isMultimodalMode()) {
    task.manual_screen_present = document.getElementById('manualScreenPresent').value;
    task.manual_screen_contains_sensitive_content = document.getElementById('manualScreenContent').value;
  }
  task.review_status = document.getElementById('reviewStatus').value;
  task.reviewer_id = document.getElementById('reviewerId').value;
  task.manual_notes = document.getElementById('manualNotes').value;
  task.reviewed = task.review_status === 'reviewed';
  updateSummary();
  renderTaskList();
}

function drawBox(box, idx, colorOverride, selectedOverride = null) {
  const selected = selectedOverride === null ? idx === state.selectedBox : selectedOverride;
  const color = colorOverride || (selected ? '#9b3d12' : '#2d6a4f');
  ctx.strokeStyle = color;
  ctx.lineWidth = selected ? 3 : 2;
  ctx.strokeRect(box.x1 * state.scale, box.y1 * state.scale, (box.x2 - box.x1) * state.scale, (box.y2 - box.y1) * state.scale);
  ctx.fillStyle = color;
  ctx.fillRect(box.x1 * state.scale, box.y1 * state.scale - 18, 42, 18);
  ctx.fillStyle = '#fff';
  ctx.font = '12px Georgia';
  ctx.fillText(String(idx + 1), box.x1 * state.scale + 6, box.y1 * state.scale - 5);
  if (selected) {
    const handles = handlePoints(box);
    ctx.fillStyle = color;
    handles.forEach(p => ctx.fillRect(p.x * state.scale - 4, p.y * state.scale - 4, 8, 8));
  }
}

function render() {
  const img = state.image;
  canvas.width = img.width * state.scale;
  canvas.height = img.height * state.scale;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  if (isMultimodalMode()) {
    currentTask().text_boxes.forEach((box, idx) => drawBox(
      box, idx, '#0066cc', state.activeBoxType === 'text' && idx === state.selectedBox
    ));
    currentTask().screen_boxes.forEach((box, idx) => drawBox(
      box, idx, '#228b22', state.activeBoxType === 'screen' && idx === state.selectedBox
    ));
  } else if (isFaceMode() || isTextMode() || isScreenMode()) {
    activeBoxes().forEach((box, idx) => drawBox(box, idx, isTextMode() ? '#0066cc' : (isScreenMode() ? '#228b22' : '#9b3d12')));
  }
  renderTaskList();
  renderBoxList();
  renderMeta();
  updateSummary();
  syncFaceCategoryFormFromTask();
  syncTextFormFromTask();
}

function handlePoints(box) {
  return [
    {name: 'nw', x: box.x1, y: box.y1},
    {name: 'ne', x: box.x2, y: box.y1},
    {name: 'sw', x: box.x1, y: box.y2},
    {name: 'se', x: box.x2, y: box.y2},
  ];
}

function pickBox(x, y) {
  const layers = isMultimodalMode()
    ? [state.activeBoxType, state.activeBoxType === 'text' ? 'screen' : 'text']
    : [state.activeBoxType];
  for (const type of layers) {
    const boxes = isMultimodalMode()
      ? (type === 'text' ? currentTask().text_boxes : currentTask().screen_boxes)
      : activeBoxes();
    for (let i = boxes.length - 1; i >= 0; i--) {
      const b = boxes[i];
      if (x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2) return {type, index: i};
    }
  }
  return null;
}

function pickHandle(x, y) {
  if (state.selectedBox === null) return null;
  const box = activeBoxes()[state.selectedBox];
  for (const handle of handlePoints(box)) {
    if (Math.abs(x - handle.x) <= 8 && Math.abs(y - handle.y) <= 8) return handle.name;
  }
  return null;
}

function clampBox(box, width, height) {
  box.x1 = Math.max(0, Math.min(box.x1, width - 1));
  box.y1 = Math.max(0, Math.min(box.y1, height - 1));
  box.x2 = Math.max(box.x1 + 1, Math.min(box.x2, width));
  box.y2 = Math.max(box.y1 + 1, Math.min(box.y2, height));
}

canvas.addEventListener('mousedown', (event) => {
  if (!canEditBoxes()) return;
  const rect = canvas.getBoundingClientRect();
  const x = Math.round((event.clientX - rect.left) / state.scale);
  const y = Math.round((event.clientY - rect.top) / state.scale);
  const handle = pickHandle(x, y);
  if (state.interactionMode === 'add') {
    const box = {x1: x, y1: y, x2: x + 1, y2: y + 1, score: '', annotator_id: 'reviewed', annotation_round: 1, condition_label: currentTask().condition_label, notes: ''};
    activeBoxes().push(box);
    state.selectedBox = activeBoxes().length - 1;
    state.interactionMode = 'drawing';
    state.dragStart = {x, y};
    render();
    return;
  }
  if (handle) {
    state.interactionMode = 'resize';
    state.activeHandle = handle;
    state.dragStart = {x, y};
    return;
  }
  const picked = pickBox(x, y);
  if (picked !== null) {
    if (isMultimodalMode()) {
      state.activeBoxType = picked.type;
      document.getElementById('textLayerBtn').classList.toggle('layer-active', picked.type === 'text');
      document.getElementById('screenLayerBtn').classList.toggle('layer-active', picked.type === 'screen');
      document.getElementById('addBoxBtn').textContent = picked.type === 'text' ? 'Add Text Box' : 'Add Screen Box';
    }
    state.selectedBox = picked.index;
    state.interactionMode = 'move';
    state.dragStart = {x, y};
    render();
    return;
  }
  state.selectedBox = null;
  render();
});

canvas.addEventListener('mousemove', (event) => {
  if (!canEditBoxes()) return;
  if (state.interactionMode === 'idle' || (state.selectedBox === null && state.interactionMode !== 'drawing')) return;
  const rect = canvas.getBoundingClientRect();
  const x = Math.round((event.clientX - rect.left) / state.scale);
  const y = Math.round((event.clientY - rect.top) / state.scale);
  const box = activeBoxes()[state.selectedBox];
  const width = state.image.width;
  const height = state.image.height;
  if (state.interactionMode === 'move' && state.dragStart) {
    const dx = x - state.dragStart.x;
    const dy = y - state.dragStart.y;
    box.x1 += dx; box.x2 += dx; box.y1 += dy; box.y2 += dy;
    clampBox(box, width, height);
    state.dragStart = {x, y};
  } else if (state.interactionMode === 'resize') {
    if (state.activeHandle.includes('n')) box.y1 = y;
    if (state.activeHandle.includes('s')) box.y2 = y;
    if (state.activeHandle.includes('w')) box.x1 = x;
    if (state.activeHandle.includes('e')) box.x2 = x;
    if (box.x2 <= box.x1) [box.x1, box.x2] = [Math.min(box.x1, box.x2 - 1), Math.max(box.x2, box.x1 + 1)];
    if (box.y2 <= box.y1) [box.y1, box.y2] = [Math.min(box.y1, box.y2 - 1), Math.max(box.y2, box.y1 + 1)];
    clampBox(box, width, height);
  } else if (state.interactionMode === 'drawing') {
    box.x1 = Math.min(state.dragStart.x, x);
    box.y1 = Math.min(state.dragStart.y, y);
    box.x2 = Math.max(state.dragStart.x + 1, x);
    box.y2 = Math.max(state.dragStart.y + 1, y);
    clampBox(box, width, height);
  }
  render();
});

window.addEventListener('mouseup', () => {
  if (!canEditBoxes()) return;
  if (state.interactionMode === 'drawing') state.interactionMode = 'idle';
  if (state.interactionMode === 'move' || state.interactionMode === 'resize') state.interactionMode = 'idle';
  state.dragStart = null;
  state.activeHandle = null;
});

document.getElementById('addBoxBtn').onclick = () => {
  if (!canEditBoxes()) return;
  state.interactionMode = 'add';
  setStatus('Add mode: drag a new box on the image.');
};

document.getElementById('textLayerBtn').onclick = () => setActiveBoxType('text');
document.getElementById('screenLayerBtn').onclick = () => setActiveBoxType('screen');

document.getElementById('deleteBoxBtn').onclick = () => {
  if (!canEditBoxes() || state.selectedBox === null) return;
  activeBoxes().splice(state.selectedBox, 1);
  state.selectedBox = null;
  render();
};

document.getElementById('markReviewedBtn').onclick = () => {
  const task = currentTask();
  task.reviewed = true;
  if (isFaceMode()) {
    task.category_review_status = 'reviewed';
    syncFaceCategoryFormFromTask();
  }
  if (isBinaryReviewMode()) {
    task.review_status = 'reviewed';
    syncTextFormFromTask();
  }
  render();
};

document.getElementById('markAllReviewedBtn').onclick = () => {
  const confirmed = window.confirm('Mark every loaded image as reviewed? This does not change boxes or category values. Use Save CSV afterwards to persist.');
  if (!confirmed) return;
  state.tasks.forEach((task) => {
    task.reviewed = true;
    if (isFaceMode()) {
      task.category_review_status = 'reviewed';
    }
    if (isBinaryReviewMode()) {
      task.review_status = 'reviewed';
    }
  });
  render();
  setStatus(`Marked all ${state.tasks.length} images as reviewed. Click Save CSV to persist.`);
};

document.getElementById('prevBtn').onclick = () => loadTask(Math.max(0, state.index - 1));
document.getElementById('nextBtn').onclick = () => loadTask(Math.min(state.tasks.length - 1, state.index + 1));

document.getElementById('saveBtn').onclick = async () => {
  if (isFaceMode()) syncTaskFromFaceCategoryForm();
  if (isBinaryReviewMode()) syncTaskFromTextForm();
  // Map the single-layer modes back to their persistent arrays.
  state.tasks.forEach(task => {
    if (isTextMode()) {
      task.text_boxes = task.boxes || [];
    } else if (isScreenMode()) {
      task.screen_boxes = task.boxes || [];
    }
  });
  const response = await fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode: state.mode, tasks: state.tasks}),
  });
  const payload = await response.json();
  setStatus(payload.message);
};

['manualTextPresent', 'manualLegibleText', 'manualScreenPresent', 'manualScreenContent', 'reviewStatus', 'reviewerId', 'manualNotes'].forEach((id) => {
  document.getElementById(id).addEventListener('change', syncTaskFromTextForm);
  document.getElementById(id).addEventListener('input', syncTaskFromTextForm);
});

[
  'faceCountCategory',
  'faceScaleCategory',
  'edgePartialFace',
  'profileOccludedFace',
  'downwardEgocentricView',
  'blurLowSharpness',
  'lowLightDim',
  'clutterLevel',
  'outdoorVehicleScene',
  'categoryReviewStatus',
  'categoryNotes',
].forEach((id) => {
  document.getElementById(id).addEventListener('change', syncTaskFromFaceCategoryForm);
  document.getElementById(id).addEventListener('input', syncTaskFromFaceCategoryForm);
});

function configureForMode() {
  const allowBoxes = canEditBoxes();
  document.getElementById('appTitle').textContent = isFaceMode()
    ? 'Face Reviewer'
    : (isMultimodalMode() ? 'Multimodal Privacy Reviewer' : (isTextMode() ? 'Text Reviewer' : 'Screen Reviewer'));
  document.querySelector('#textReviewPanel h3').textContent = isMultimodalMode()
    ? 'Text and Screen Review'
    : (isTextMode() ? 'Text Review' : 'Screen Review');
  document.querySelector('label[for="manualTextPresent"]').textContent = isScreenMode()
    ? 'Visible screen present'
    : 'Visible text present';
  document.querySelector('label[for="manualLegibleText"]').textContent = isScreenMode()
    ? 'Readable or potentially sensitive screen content'
    : 'Contains legible text';
  document.getElementById('legendBox').classList.toggle('hidden', !allowBoxes);
  document.getElementById('faceCategoryPanel').classList.toggle('hidden', !isFaceMode());
  document.getElementById('textReviewPanel').classList.toggle('hidden', !isBinaryReviewMode());
  document.getElementById('multimodalLayerControls').classList.toggle('hidden', !isMultimodalMode());
  document.querySelector('label[for="manualScreenPresent"]').classList.toggle('hidden', !isMultimodalMode());
  document.getElementById('manualScreenPresent').classList.toggle('hidden', !isMultimodalMode());
  document.querySelector('label[for="manualScreenContent"]').classList.toggle('hidden', !isMultimodalMode());
  document.getElementById('manualScreenContent').classList.toggle('hidden', !isMultimodalMode());
  document.getElementById('addBoxBtn').disabled = !allowBoxes;
  document.getElementById('deleteBoxBtn').disabled = !allowBoxes;
  if (isMultimodalMode()) {
    document.getElementById('legendBox').innerHTML = `
      <strong style="color:#0066cc">Blue: text regions</strong> ·
      <strong style="color:#228b22">Green: screen regions</strong>.<br>
      Choose a layer, then click a box to move/resize it or use Add Box to draw a new region.
    `;
    setActiveBoxType('text');
  } else if (isTextMode() || isScreenMode()) {
    const color = isTextMode() ? '#0066cc' : '#228b22';
    const label = isTextMode() ? 'Text boxes' : 'Screen boxes';
    document.getElementById('legendBox').innerHTML = `
      <strong style="color:${color}">${label}</strong>: click a box to select it,
      drag inside to move, drag a corner to resize, or use Add Box to draw a new region.
    `;
  }
}

async function loadTask(index) {
  if (isFaceMode() && state.image.src) syncTaskFromFaceCategoryForm();
  if (isBinaryReviewMode()) syncTaskFromTextForm();
  state.index = index;
  state.selectedBox = null;
  const task = currentTask();
  if (isTextMode()) {
    task.boxes = task.text_boxes || [];
  } else if (isScreenMode()) {
    task.boxes = task.screen_boxes || [];
  }
  const imageUrl = '/api/image?path=' + encodeURIComponent(task.image_path);
  await new Promise((resolve, reject) => {
    state.image.onload = () => resolve();
    state.image.onerror = reject;
    state.image.src = imageUrl;
  });
  const maxWidth = Math.min(window.innerWidth - 440, 1280);
  state.scale = Math.min(1, maxWidth / state.image.width);
  render();
  setStatus(`Loaded ${task.relative_path}`);
}

async function initialise() {
  const response = await fetch('/api/tasks');
  const payload = await response.json();
  state.mode = payload.mode;
  state.tasks = payload.tasks;
  configureForMode();
  await loadTask(0);
}

initialise().catch(error => {
  console.error(error);
  setStatus('Failed to load reviewer data.');
});
</script>
</body>
</html>
"""


def load_face_tasks() -> list[dict[str, Any]]:
    """Load face annotation tasks and existing proposals."""
    tasks: dict[str, dict[str, Any]] = {}
    compact_manifest_rows: dict[str, dict[str, str]] = {}
    category_lookup: dict[str, dict[str, str]] = {}
    if FACE_CATEGORY_CSV_PATH.exists():
        with FACE_CATEGORY_CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = row.get("image_id") or row.get("relative_path") or ""
                if image_id:
                    category_lookup[image_id] = row
                    compact_manifest_rows[image_id] = row

    with FACE_TASKS_CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relative_path = row["relative_path"]
            category = category_lookup.get(relative_path, row)
            image_path = row.get("image_path") or row.get("raw_path") or str(
                PROJECT_ROOT / "data" / "castle2024" / "raw" / relative_path
            )
            tasks[relative_path] = {
                "relative_path": relative_path,
                "image_path": image_path,
                "camera_stream_id": row.get("camera_stream_id") or row.get("participant_id", ""),
                "day_id": row.get("day_id") or row.get("day_or_session_id", ""),
                "view_type": row.get("view_type") or "egocentric",
                "participant_id": row.get("participant_id", ""),
                "condition_label": row.get("condition_label") or category.get("condition_label", ""),
                "face_count_category": category.get("face_count_category", ""),
                "face_scale_category": category.get("face_scale_category", ""),
                "edge_partial_face": category.get("edge_partial_face", ""),
                "profile_occluded_face": category.get("profile_occluded_face", ""),
                "downward_egocentric_view": category.get("downward_egocentric_view", ""),
                "blur_low_sharpness": category.get("blur_low_sharpness", ""),
                "low_light_dim": category.get("low_light_dim", ""),
                "clutter_level": category.get("clutter_level", ""),
                "text_screen_risk": category.get("text_screen_risk", ""),
                "outdoor_vehicle_scene": category.get("outdoor_vehicle_scene", ""),
                "category_review_status": category.get("category_review_status", "pending"),
                "category_notes": category.get("category_notes", ""),
                "reviewed": category.get("manual_review_status") in {"yes", "reviewed"},
                "boxes": [],
            }

    proposal_path = FACE_OUTPUT_CSV_PATH if FACE_OUTPUT_CSV_PATH.exists() else FACE_PROPOSALS_CSV_PATH
    with proposal_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "reviewed_face_boxes_json" in row:
                task = tasks.get(row.get("image_id") or row.get("relative_path", ""))
                if task is None:
                    continue
                try:
                    boxes = json.loads(row.get("reviewed_face_boxes_json") or "[]")
                except json.JSONDecodeError:
                    boxes = []
                for box in boxes:
                    task["boxes"].append(
                        {
                            "x1": int(box["x1"]),
                            "y1": int(box["y1"]),
                            "x2": int(box["x2"]),
                            "y2": int(box["y2"]),
                            "score": box.get("score", ""),
                            "annotator_id": box.get("annotator_id", "reviewed"),
                            "annotation_round": int(box.get("annotation_round", 1) or 1),
                            "condition_label": box.get("condition_label", task["condition_label"]),
                            "notes": box.get("notes", ""),
                        }
                    )
                continue
            task = tasks.get(row["image_id"])
            if task is None:
                continue
            task["boxes"].append(
                {
                    "x1": int(row["x1"]),
                    "y1": int(row["y1"]),
                    "x2": int(row["x2"]),
                    "y2": int(row["y2"]),
                    "score": row.get("score", ""),
                    "annotator_id": row.get("annotator_id", "reviewed"),
                    "annotation_round": int(row.get("annotation_round", 1) or 1),
                    "condition_label": row.get("condition_label", task["condition_label"]),
                    "notes": row.get("notes", ""),
                }
            )

    reviewed_lookup: dict[str, bool] = {}
    if FACE_REVIEW_STATUS_CSV_PATH.exists():
        with FACE_REVIEW_STATUS_CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = row.get("image_id") or row.get("relative_path") or ""
                if not image_id:
                    continue
                reviewed_lookup[image_id] = row.get("reviewed") == "yes" or row.get("manual_review_status") in {"yes", "reviewed"}
    for task in tasks.values():
        task["reviewed"] = reviewed_lookup.get(task["relative_path"], task["reviewed"])
    return list(tasks.values())


def load_text_tasks() -> list[dict[str, Any]]:
    """Load text presence review rows."""
    text_pack_root, text_tasks_csv_path, _ = resolve_text_review_paths()
    tasks: list[dict[str, Any]] = []
    with text_tasks_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relative_path = row.get("relative_path") or row.get("image_id") or ""
            text_region_count = int(row.get("text_region_count") or 0)
            review_status = row.get("text_review_status") or "pending"
            image_path = row.get("image_path") or str(PROJECT_ROOT / "data" / "castle2024" / "raw" / relative_path)
            try:
                text_boxes = json.loads(row.get("text_boxes_json") or "[]")
                screen_boxes = json.loads(row.get("screen_boxes_json") or "[]")
            except json.JSONDecodeError:
                text_boxes, screen_boxes = [], []
            tasks.append(
                {
                    "relative_path": relative_path,
                    "image_path": image_path if Path(image_path).is_absolute() else str(text_pack_root / "images" / image_path),
                    "camera_stream_id": row.get("camera_stream_id", ""),
                    "day_id": row.get("day_id", ""),
                    "view_type": row.get("view_type", ""),
                    "condition_label": row.get("condition_label") or row.get("text_legibility_subgroup", ""),
                    "visible_text_flag": (
                        row.get("visible_text_flag", "").strip().lower() == "true"
                        if row.get("visible_text_flag", "") != ""
                        else text_region_count > 0
                    ),
                    "sample_bucket": row.get("sample_bucket") or row.get("text_legibility_subgroup", ""),
                    "manual_text_present": row.get("manual_text_present") or ("yes" if text_region_count > 0 else "no"),
                    "manual_contains_legible_text": row.get("manual_contains_legible_text") or (
                        "yes" if row.get("text_legibility_subgroup") == "clearly_visible_or_legible_text_candidate" else "no"
                    ),
                    "review_status": review_status,
                    "reviewer_id": row.get("reviewer_id", ""),
                    "manual_notes": row.get("manual_notes", ""),
                    "reviewed": review_status == "reviewed",
                    "boxes": text_boxes,
                    "text_boxes": text_boxes,
                    "screen_boxes": screen_boxes,
                }
            )
    return tasks


def load_screen_tasks() -> list[dict[str, Any]]:
    """Load screen presence review rows."""
    screen_pack_root, screen_tasks_csv_path, _ = resolve_text_review_paths()
    tasks: list[dict[str, Any]] = []
    with screen_tasks_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relative_path = row.get("relative_path") or row.get("image_id") or ""
            screen_region_count = int(row.get("screen_region_count") or 0)
            review_status = row.get("screen_review_status") or "pending"
            image_path = row.get("image_path") or str(PROJECT_ROOT / "data" / "castle2024" / "raw" / relative_path)
            try:
                text_boxes = json.loads(row.get("text_boxes_json") or "[]")
                screen_boxes = json.loads(row.get("screen_boxes_json") or "[]")
            except json.JSONDecodeError:
                text_boxes, screen_boxes = [], []
            tasks.append(
                {
                    "relative_path": relative_path,
                    "image_path": image_path if Path(image_path).is_absolute() else str(screen_pack_root / "images" / image_path),
                    "camera_stream_id": row.get("camera_stream_id", ""),
                    "day_id": row.get("day_id", ""),
                    "view_type": row.get("view_type", ""),
                    "condition_label": row.get("condition_label") or row.get("screen_subgroup", ""),
                    "visible_screen_flag": (
                        row.get("visible_screen_flag", "").strip().lower() == "true"
                        if row.get("visible_screen_flag", "") != ""
                        else screen_region_count > 0
                    ),
                    "sample_bucket": row.get("sample_bucket") or row.get("screen_subgroup", ""),
                    "manual_screen_present": row.get("manual_screen_present") or ("yes" if screen_region_count > 0 else "no"),
                    "manual_screen_contains_sensitive_content": row.get(
                        "manual_screen_contains_sensitive_content", ""
                    ),
                    "review_status": review_status,
                    "reviewer_id": row.get("reviewer_id", ""),
                    "manual_notes": row.get("manual_notes", ""),
                    "reviewed": review_status == "reviewed",
                    "boxes": screen_boxes,
                    "text_boxes": text_boxes,
                    "screen_boxes": screen_boxes,
                }
            )
    return tasks


def load_multimodal_tasks() -> list[dict[str, Any]]:
    """Load text and screen annotations into one combined review task."""
    pack_root, tasks_csv_path, _ = resolve_text_review_paths()
    tasks: list[dict[str, Any]] = []
    with tasks_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relative_path = row.get("relative_path") or row.get("image_id") or ""
            image_path = row.get("image_path") or str(
                PROJECT_ROOT / "data" / "castle2024" / "raw" / relative_path
            )
            try:
                text_boxes = json.loads(row.get("text_boxes_json") or "[]")
                screen_boxes = json.loads(row.get("screen_boxes_json") or "[]")
            except json.JSONDecodeError:
                text_boxes, screen_boxes = [], []
            review_status = (
                "reviewed" if row.get("reviewed") in {"yes", "reviewed"} else "pending"
            )
            tasks.append(
                {
                    "relative_path": relative_path,
                    "image_path": (
                        image_path
                        if Path(image_path).is_absolute()
                        else str(pack_root / "images" / image_path)
                    ),
                    "camera_stream_id": row.get("camera_stream_id")
                    or row.get("participant_id", ""),
                    "day_id": row.get("day_id", ""),
                    "view_type": row.get("view_type", ""),
                    "condition_label": "multimodal_text_screen_review",
                    "sample_bucket": "multimodal_250",
                    "manual_text_present": row.get("manual_text_present")
                    or ("yes" if text_boxes else "no"),
                    "manual_contains_legible_text": row.get(
                        "manual_contains_legible_text"
                    )
                    or "no",
                    "manual_screen_present": row.get("manual_screen_present")
                    or ("yes" if screen_boxes else "no"),
                    "manual_screen_contains_sensitive_content": row.get(
                        "manual_screen_contains_sensitive_content"
                    )
                    or "no",
                    "review_status": review_status,
                    "reviewer_id": row.get("reviewer_id", ""),
                    "manual_notes": row.get("manual_notes", ""),
                    "reviewed": review_status == "reviewed",
                    "text_boxes": text_boxes,
                    "screen_boxes": screen_boxes,
                }
            )
    return tasks


def save_face_tasks(tasks: list[dict[str, Any]]) -> Path:
    """Persist reviewed face tasks to canonical CSV outputs."""
    if FACE_OUTPUT_CSV_PATH.exists():
        with FACE_OUTPUT_CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            manifest_rows = list(reader)
    else:
        fieldnames = []
        manifest_rows = []
    if manifest_rows and "reviewed_face_boxes_json" in fieldnames:
        task_lookup = {task["relative_path"]: task for task in tasks}
        required_fields = [
            "image_id",
            "manual_review_status",
            "manual_face_count",
            "review_method",
            "reviewed_face_boxes_json",
            "face_count_category",
            "face_scale_category",
            "edge_partial_face",
            "profile_occluded_face",
            "downward_egocentric_view",
            "blur_low_sharpness",
            "low_light_dim",
            "clutter_level",
            "text_screen_risk",
            "outdoor_vehicle_scene",
            "category_review_status",
            "condition_label",
            "category_notes",
        ]
        for field in required_fields:
            if field not in fieldnames:
                fieldnames.append(field)
        updated_rows: list[dict[str, Any]] = []
        for row in manifest_rows:
            image_id = row.get("image_id") or row.get("relative_path") or ""
            task = task_lookup.get(image_id)
            if task is None:
                updated_rows.append(row)
                continue
            boxes = [
                {
                    "x1": int(box["x1"]),
                    "y1": int(box["y1"]),
                    "x2": int(box["x2"]),
                    "y2": int(box["y2"]),
                }
                for box in task["boxes"]
            ]
            row.update(
                {
                    "image_id": task["relative_path"],
                    "manual_review_status": "yes" if task.get("reviewed") else "no",
                    "manual_face_count": str(len(task["boxes"])),
                    "review_method": row.get("review_method") or "manual reviewer-app verification by the project reviewers",
                    "reviewed_face_boxes_json": json.dumps(boxes, separators=(",", ":")),
                    "face_count_category": task.get("face_count_category", ""),
                    "face_scale_category": task.get("face_scale_category", ""),
                    "edge_partial_face": task.get("edge_partial_face", ""),
                    "profile_occluded_face": task.get("profile_occluded_face", ""),
                    "downward_egocentric_view": task.get("downward_egocentric_view", ""),
                    "blur_low_sharpness": task.get("blur_low_sharpness", ""),
                    "low_light_dim": task.get("low_light_dim", ""),
                    "clutter_level": task.get("clutter_level", ""),
                    "text_screen_risk": task.get("text_screen_risk", ""),
                    "outdoor_vehicle_scene": task.get("outdoor_vehicle_scene", ""),
                    "category_review_status": task.get("category_review_status", "pending"),
                    "condition_label": task.get("condition_label", ""),
                    "category_notes": task.get("category_notes", ""),
                }
            )
            updated_rows.append(row)
        FACE_OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=FACE_OUTPUT_CSV_PATH.parent, delete=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
            temp_path = Path(handle.name)
        temp_path.replace(FACE_OUTPUT_CSV_PATH)
        return FACE_OUTPUT_CSV_PATH

    rows: list[dict[str, Any]] = []
    for task in tasks:
        for box in task["boxes"]:
            rows.append(
                {
                    "image_id": task["relative_path"],
                    "x1": int(box["x1"]),
                    "y1": int(box["y1"]),
                    "x2": int(box["x2"]),
                    "y2": int(box["y2"]),
                    "annotator_id": box.get("annotator_id") or "reviewed",
                    "annotation_round": int(box.get("annotation_round") or 1),
                    "condition_label": box.get("condition_label") or task["condition_label"],
                    "notes": box.get("notes", ""),
                }
            )
    FACE_OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=FACE_OUTPUT_CSV_PATH.parent, delete=False) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "annotator_id",
                "annotation_round",
                "condition_label",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(FACE_OUTPUT_CSV_PATH)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=FACE_REVIEW_STATUS_CSV_PATH.parent,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "reviewed",
                "box_count",
                "annotator_id",
                "condition_label",
            ],
        )
        writer.writeheader()
        for task in tasks:
            box_annotators = sorted({str(box.get("annotator_id") or "reviewed") for box in task["boxes"]})
            writer.writerow(
                {
                    "image_id": task["relative_path"],
                    "reviewed": "yes" if task.get("reviewed") else "no",
                    "box_count": len(task["boxes"]),
                    "annotator_id": "|".join(box_annotators) if box_annotators else "",
                    "condition_label": task["condition_label"],
                }
            )
        temp_path = Path(handle.name)
    temp_path.replace(FACE_REVIEW_STATUS_CSV_PATH)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=FACE_CATEGORY_CSV_PATH.parent,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "face_count_category",
                "face_scale_category",
                "edge_partial_face",
                "profile_occluded_face",
                "downward_egocentric_view",
                "blur_low_sharpness",
                "low_light_dim",
                "clutter_level",
                "text_screen_risk",
                "outdoor_vehicle_scene",
                "category_review_status",
                "condition_label",
                "category_notes",
            ],
        )
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "image_id": task["relative_path"],
                    "face_count_category": task.get("face_count_category", ""),
                    "face_scale_category": task.get("face_scale_category", ""),
                    "edge_partial_face": task.get("edge_partial_face", ""),
                    "profile_occluded_face": task.get("profile_occluded_face", ""),
                    "downward_egocentric_view": task.get("downward_egocentric_view", ""),
                    "blur_low_sharpness": task.get("blur_low_sharpness", ""),
                    "low_light_dim": task.get("low_light_dim", ""),
                    "clutter_level": task.get("clutter_level", ""),
                    "text_screen_risk": task.get("text_screen_risk", ""),
                    "outdoor_vehicle_scene": task.get("outdoor_vehicle_scene", ""),
                    "category_review_status": task.get("category_review_status", "pending"),
                    "condition_label": task.get("condition_label", ""),
                    "category_notes": task.get("category_notes", ""),
                }
            )
        temp_path = Path(handle.name)
    temp_path.replace(FACE_CATEGORY_CSV_PATH)
    return FACE_OUTPUT_CSV_PATH


def save_multimodal_tasks(tasks: list[dict[str, Any]], box_type: str) -> Path:
    """Update one multimodal review layer without discarding the other layer."""
    _, source_path, output_path = resolve_text_review_paths()
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for required in (
        "manual_text_present",
        "manual_contains_legible_text",
        "manual_screen_present",
        "manual_screen_contains_sensitive_content",
        "text_review_status",
        "screen_review_status",
    ):
        if required not in fieldnames:
            fieldnames.append(required)

    task_lookup = {task["relative_path"]: task for task in tasks}
    for row in rows:
        image_id = row.get("image_id") or row.get("relative_path") or ""
        task = task_lookup.get(image_id)
        if task is None:
            continue
        boxes = task.get(f"{box_type}_boxes", [])
        row[f"{box_type}_boxes_json"] = json.dumps(boxes, separators=(",", ":"))
        row[f"{box_type}_region_count"] = str(len(boxes))
        row[f"{box_type}_review_status"] = task.get("review_status", "pending")
        if box_type == "text":
            row["manual_text_present"] = task.get("manual_text_present", "")
            row["manual_contains_legible_text"] = task.get(
                "manual_contains_legible_text", ""
            )
            if row["manual_text_present"] == "no":
                row["text_legibility_subgroup"] = "no_detected_text"
            elif row["manual_contains_legible_text"] == "yes":
                row["text_legibility_subgroup"] = (
                    "clearly_visible_or_legible_text_candidate"
                )
            else:
                row["text_legibility_subgroup"] = (
                    "symbols_short_or_low_legibility_text_candidate"
                )
        else:
            row["manual_screen_present"] = task.get("manual_screen_present", "")
            row["manual_screen_contains_sensitive_content"] = task.get(
                "manual_screen_contains_sensitive_content", ""
            )
            row["screen_subgroup"] = (
                "screen_present_candidate"
                if row["manual_screen_present"] == "yes"
                else "no_detected_screen"
            )
        both_reviewed = all(
            row.get(f"{kind}_review_status") == "reviewed" for kind in ("text", "screen")
        )
        row["reviewed"] = "yes" if both_reviewed else "pending"
        row["review_method"] = (
            "manual reviewer-app verification by the project reviewers"
            if both_reviewed
            else "pending reviewer-app verification"
        )

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=output_path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)
    return output_path


def save_text_tasks(tasks: list[dict[str, Any]]) -> Path:
    """Persist text boxes while preserving screen boxes and evidence fields."""
    return save_multimodal_tasks(tasks, "text")


def save_screen_tasks(tasks: list[dict[str, Any]]) -> Path:
    """Persist screen boxes while preserving text boxes and evidence fields."""
    return save_multimodal_tasks(tasks, "screen")


def save_combined_multimodal_tasks(tasks: list[dict[str, Any]]) -> Path:
    """Persist both multimodal layers and their shared review state atomically."""
    _, source_path, output_path = resolve_text_review_paths()
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    required_fields = (
        "text_boxes_json",
        "screen_boxes_json",
        "manual_text_present",
        "manual_contains_legible_text",
        "manual_screen_present",
        "manual_screen_contains_sensitive_content",
        "text_review_status",
        "screen_review_status",
        "reviewer_id",
        "manual_notes",
    )
    for field in required_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    task_lookup = {task["relative_path"]: task for task in tasks}
    for row in rows:
        image_id = row.get("image_id") or row.get("relative_path") or ""
        task = task_lookup.get(image_id)
        if task is None:
            continue
        text_boxes = task.get("text_boxes", [])
        screen_boxes = task.get("screen_boxes", [])
        review_status = task.get("review_status", "pending")
        row.update(
            {
                "text_boxes_json": json.dumps(text_boxes, separators=(",", ":")),
                "screen_boxes_json": json.dumps(screen_boxes, separators=(",", ":")),
                "text_region_count": str(len(text_boxes)),
                "screen_region_count": str(len(screen_boxes)),
                "manual_text_present": task.get("manual_text_present", ""),
                "manual_contains_legible_text": task.get(
                    "manual_contains_legible_text", ""
                ),
                "manual_screen_present": task.get("manual_screen_present", ""),
                "manual_screen_contains_sensitive_content": task.get(
                    "manual_screen_contains_sensitive_content", ""
                ),
                "text_review_status": review_status,
                "screen_review_status": review_status,
                "reviewed": "yes" if review_status == "reviewed" else "pending",
                "review_method": (
                    "manual reviewer-app verification by the project reviewers"
                    if review_status == "reviewed"
                    else "pending reviewer-app verification"
                ),
                "reviewer_id": task.get("reviewer_id", ""),
                "manual_notes": task.get("manual_notes", ""),
            }
        )
        if row["manual_text_present"] == "no":
            row["text_legibility_subgroup"] = "no_detected_text"
        elif row["manual_contains_legible_text"] == "yes":
            row["text_legibility_subgroup"] = (
                "clearly_visible_or_legible_text_candidate"
            )
        else:
            row["text_legibility_subgroup"] = (
                "symbols_short_or_low_legibility_text_candidate"
            )
        row["screen_subgroup"] = (
            "screen_present_candidate"
            if row["manual_screen_present"] == "yes"
            else "no_detected_screen"
        )

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=output_path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)
    return output_path


def load_tasks(mode: str) -> list[dict[str, Any]]:
    if mode == "face_boxes":
        return load_face_tasks()
    if mode == "text_presence":
        return load_text_tasks()
    if mode == "screen_presence":
        return load_screen_tasks()
    if mode == "multimodal_presence":
        return load_multimodal_tasks()
    raise ValueError(f"Unsupported reviewer mode: {mode}")


def save_tasks(mode: str, tasks: list[dict[str, Any]]) -> Path:
    if mode == "face_boxes":
        return save_face_tasks(tasks)
    if mode == "text_presence":
        return save_text_tasks(tasks)
    if mode == "screen_presence":
        return save_screen_tasks(tasks)
    if mode == "multimodal_presence":
        return save_combined_multimodal_tasks(tasks)
    raise ValueError(f"Unsupported reviewer mode: {mode}")


class ReviewerHandler(BaseHTTPRequestHandler):
    """Serve the browser UI and a small JSON/file API for review workflows."""

    reviewer_mode = "face_boxes"

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/index"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/tasks"):
            self._send_json({"mode": self.reviewer_mode, "tasks": load_tasks(self.reviewer_mode)})
            return

        if self.path.startswith("/api/image"):
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            relative = query.get("path", [""])[0]
            image_path = PROJECT_ROOT / relative
            if not image_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
                return
            body = image_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/webp")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        tasks = payload.get("tasks", [])
        mode = payload.get("mode", self.reviewer_mode)
        output_path = save_tasks(mode, tasks)
        self._send_json({"message": f"Saved reviewed data to {output_path.relative_to(PROJECT_ROOT)}"})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def run_server(host: str = "127.0.0.1", port: int = 8765, mode: str = "face_boxes") -> None:
    """Start the reviewer server in the requested mode."""
    if mode not in {
        "face_boxes",
        "text_presence",
        "screen_presence",
        "multimodal_presence",
    }:
        raise ValueError("unsupported reviewer mode")

    class ModeAwareReviewerHandler(ReviewerHandler):
        reviewer_mode = mode

    server = ThreadingHTTPServer((host, port), ModeAwareReviewerHandler)
    label = {
        "face_boxes": "Face annotation reviewer",
        "text_presence": "Text validation reviewer",
        "screen_presence": "Screen validation reviewer",
        "multimodal_presence": "Multimodal privacy reviewer",
    }[mode]
    print(f"{label} running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
