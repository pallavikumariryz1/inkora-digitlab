"""Build an interactive, browser-side forward-pass visualisation from real CNN tensors."""
from __future__ import annotations

import json
import numpy as np


def _normalise_map(array: np.ndarray) -> list[list[float]]:
    a = np.asarray(array, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        a = (a - lo) / (hi - lo)
    else:
        a = np.zeros_like(a)
    return np.round(a, 3).tolist()


def build_trace(model, image: np.ndarray, probabilities: np.ndarray) -> dict:
    """Extract real intermediate activations and learned dense weights for one input."""
    import tensorflow as tf

    x = (np.asarray(image, dtype=np.float32) / 255.0).reshape(1, 28, 28, 1)
    symbolic = tf.keras.Input(shape=(28, 28, 1), name="inkora_trace_input")
    tensor = symbolic
    wanted = {"feature_maps_1", "pool_1", "feature_maps_2", "pool_2", "learned_features"}
    outputs = {}
    for layer in model.layers:
        tensor = layer(tensor, training=False)
        if layer.name in wanted:
            outputs[layer.name] = tensor
    missing = wanted.difference(outputs)
    if missing:
        raise ValueError(f"The trained model is missing expected layers: {sorted(missing)}")
    extractor = tf.keras.Model(symbolic, [outputs[n] for n in [
        "feature_maps_1", "pool_1", "feature_maps_2", "pool_2", "learned_features"
    ]])
    conv1, pool1, conv2, pool2, dense = [np.asarray(v)[0] for v in extractor(x, training=False)]

    # Pick the four channels with the strongest mean activation, for a readable
    # display. The values are taken directly from this model's forward pass.
    def selected_channels(maps):
        scores = np.mean(np.maximum(maps, 0), axis=(0, 1))
        indices = np.argsort(scores)[-4:][::-1]
        return [{"channel": int(i + 1), "map": _normalise_map(maps[:, :, i])} for i in indices]

    first_conv_kernel = np.asarray(model.get_layer("feature_maps_1").get_weights()[0][:, :, 0, 0])
    kernel_scale = float(np.max(np.abs(first_conv_kernel))) or 1.0
    dense_layer = model.get_layer("learned_features")
    output_layer = model.get_layer("digit_probabilities")
    dense_kernel = np.asarray(dense_layer.get_weights()[0])
    output_kernel = np.asarray(output_layer.get_weights()[0])
    # 18 representative hidden neurons; their output edges use the actual
    # learned 128x10 dense-layer weights (not random illustrative connections).
    hidden_indices = np.linspace(0, 127, 18, dtype=int)
    selected_weights = output_kernel[hidden_indices, :]
    max_weight = float(np.max(np.abs(selected_weights))) or 1.0

    return {
        "input": _normalise_map(x[0, :, :, 0]),
        "kernel": np.round(first_conv_kernel / kernel_scale, 3).tolist(),
        "conv1": selected_channels(conv1),
        "pool1": _normalise_map(np.mean(pool1, axis=2)),
        "conv2": selected_channels(conv2),
        "pool2": _normalise_map(np.mean(pool2, axis=2)),
        "dense": np.round(dense[hidden_indices], 4).tolist(),
        "dense_indices": (hidden_indices + 1).tolist(),
        "weights": np.round(selected_weights / max_weight, 4).tolist(),
        "probabilities": np.round(np.asarray(probabilities, dtype=np.float32), 5).tolist(),
        "predicted": int(np.argmax(probabilities)),
        "dense_neuron_count": int(dense.shape[0]),
        "conv1_shape": list(conv1.shape),
        "conv2_shape": list(conv2.shape),
        "flatten_size": int(np.prod(pool2.shape)),
    }


def render_forward_pass(trace: dict) -> str:
    """Return a self-contained HTML canvas animation driven by actual model data."""
    payload = json.dumps(trace, separators=(",", ":")).replace("</", "<\\/")
    return r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}body{margin:0;background:#0d1018;color:#f2f1ec;font-family:Inter,Arial,sans-serif}.wrap{padding:14px;background:radial-gradient(ellipse at 50% 0,#202a48,#0d1018 68%);border:1px solid #8f794d;border-radius:16px}.top{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}.eyebrow{font-size:11px;letter-spacing:2px;color:#d6b66b;font-weight:800}.headline{font-size:20px;font-weight:800;margin:5px 0}.sub{font-size:12px;color:#b8bfd0}.stage{margin:12px 0 8px;color:#e1c477;font-size:13px;font-weight:700;min-height:18px}canvas{width:100%;height:auto;display:block;background:#090b10;border-radius:10px;border:1px solid #30384c}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px}.controls button{background:#28365f;color:#fff;border:1px solid #b89b5d;padding:9px 15px;border-radius:9px;font-weight:700;cursor:pointer}.controls button:hover{background:#3c4b79}.controls input{accent-color:#d5b76f}.hint{font-size:11px;color:#aeb5c5;margin-left:auto}.legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:9px;font-size:11px;color:#c5cada}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.gold{color:#e0c477}.stat{font-size:12px;color:#dce1eb;margin-top:7px}
</style></head><body><div class="wrap"><div class="top"><div><div class="eyebrow">INKORA · LIVE FORWARD PASS</div><div class="headline">Watch a digit travel through your CNN</div><div class="sub">The input, activations, weights and class scores below come from this trained model.</div></div><div class="eyebrow">ACTUAL MODEL · INFERENCE MODE</div></div><div class="stage" id="stage">Ready · press Play to begin</div><canvas id="viz" width="1120" height="570"></canvas><div class="controls"><button id="play">▶ Play</button><button id="step">Step forward →</button><button id="reset">↺ Replay</button><label class="sub">Speed <input id="speed" type="range" min="0.5" max="2.5" step="0.25" value="1"></label><span class="hint">Pause at any point and inspect the layer.</span></div><div class="legend"><span><i class="dot" style="background:#46c5e0"></i>positive activation</span><span><i class="dot" style="background:#e66b55"></i>negative / inhibitory weight</span><span><i class="dot" style="background:#e1c477"></i>signal in motion</span></div><div class="stat" id="stat">Model: 28×28×1 → Conv(32) → Pool → Conv(64) → Pool → Flatten → Dense(128) → Softmax(10)</div></div>
<script>
const D=__PAYLOAD__;
const canvas=document.getElementById('viz'),ctx=canvas.getContext('2d');let phase=0,playing=false,last=0,elapsed=0,speed=1,raf=null;
const steps=['Input pixels · 28 × 28 × 1','Convolution 1 · 32 learned filters · 28 × 28 × 32','Max pooling 1 · 14 × 14 × 32','Convolution 2 · 64 learned filters · 14 × 14 × 64','Max pooling 2 · 7 × 7 × 64','Flatten · 3,136 numerical features','Dense layer · 128 learned neurons (18 shown)','Output · 10 softmax class probabilities'];
const durations=[2.2,2.8,1.8,2.8,1.8,1.7,3.2,2.4];let total=durations.reduce((a,b)=>a+b,0);
function rounded(x,y,w,h,r,fill,stroke){ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fillStyle=fill;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1.4;ctx.stroke()}}
function label(t,x,y,size=13,color='#d8dce7',align='center'){ctx.fillStyle=color;ctx.font=`${size}px Arial`;ctx.textAlign=align;ctx.fillText(t,x,y)}
function mapTile(map,x,y,size,title){label(title,x+size/2,y-10,12,'#e2c477');let n=map.length;for(let r=0;r<n;r++)for(let c=0;c<n;c++){let v=map[r][c];ctx.fillStyle=`rgba(75,194,220,${.08+.92*v})`;ctx.fillRect(x+c*size/n,y+r*size/n,size/n+.4,size/n+.4)}ctx.strokeStyle='#77839b';ctx.strokeRect(x,y,size,size)}
function gridInput(x,y,size,active,local){let n=28,cell=size/n;for(let r=0;r<n;r++)for(let c=0;c<n;c++){let v=D.input[r][c];ctx.fillStyle=`rgb(${Math.round(12+v*238)},${Math.round(15+v*231)},${Math.round(22+v*215)})`;ctx.fillRect(x+c*cell,y+r*cell,cell-.6,cell-.6)}ctx.strokeStyle='#4abbd0';ctx.strokeRect(x,y,size,size);if(active){let t=Math.max(0,Math.min(.999,local));let scan=Math.floor(t*676),kx=scan%26,ky=Math.floor(scan/26);ctx.strokeStyle='#e1c477';ctx.lineWidth=2.2;ctx.strokeRect(x+kx*cell,y+ky*cell,3*cell,3*cell);ctx.fillStyle='rgba(225,196,119,.16)';ctx.fillRect(x+kx*cell,y+ky*cell,3*cell,3*cell);for(let r=0;r<3;r++)for(let c=0;c<3;c++){let w=D.kernel[r][c];ctx.fillStyle=w>=0?`rgba(55,190,220,${.25+Math.abs(w)*.5})`:`rgba(230,107,85,${.25+Math.abs(w)*.5})`;ctx.fillRect(x+kx*cell+c*cell,y+ky*cell+r*cell,cell-.7,cell-.7)}}}
function node(x,y,r,a,glow){ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=`rgba(226,235,246,${.12+.88*a})`;ctx.fill();ctx.strokeStyle='#d9e2ef';ctx.lineWidth=1.2;ctx.stroke();if(glow){ctx.shadowColor='#e1c477';ctx.shadowBlur=15;ctx.beginPath();ctx.arc(x,y,r+2,0,Math.PI*2);ctx.strokeStyle='#e1c477';ctx.stroke();ctx.shadowBlur=0}}
function drawDense(progress){let xs=[735,900],ys=Array.from({length:18},(_,i)=>76+i*24);let weights=D.weights,acts=D.dense;for(let i=0;i<18;i++)for(let j=0;j<10;j++){let w=weights[i][j],x1=xs[0],y1=ys[i],x2=xs[1]+18,y2=105+j*39;ctx.beginPath();ctx.moveTo(x1+7,y1);ctx.lineTo(x2-8,y2);ctx.strokeStyle=w>=0?`rgba(55,190,220,${.10+Math.abs(w)*.48})`:`rgba(226,93,75,${.10+Math.abs(w)*.48})`;ctx.lineWidth=.5+Math.abs(w)*1.3;ctx.stroke();if(progress>.12){let t=(progress*.8+(i*10+j)*.037)%1;let sx=x1+7+(x2-x1-15)*t,sy=y1+(y2-y1)*t;ctx.beginPath();ctx.arc(sx,sy,2.2+Math.abs(w)*1.5,0,Math.PI*2);ctx.fillStyle='#e1c477';ctx.shadowColor='#e1c477';ctx.shadowBlur=8;ctx.fill();ctx.shadowBlur=0}}
for(let i=0;i<18;i++){let a=Math.max(0,Math.min(1,acts[i]));node(xs[0],ys[i],8,a,progress>.15&&Math.floor(progress*18)===i)}for(let j=0;j<10;j++){let a=D.probabilities[j];node(xs[1]+18,105+j*39,10,a,j===D.predicted&&progress>.6);label(String(j),xs[1]+42,110+j*39,15,j===D.predicted?'#f1cf69':'#d8dce7','left');rounded(970,96+j*39,105,10,5,'#252d40');rounded(970,96+j*39,105*a,10,5,j===D.predicted?'#e1c477':'#4bbbd1')}
label('DENSE · 128',xs[0],34,12,'#e1c477');label('SOFTMAX · 10',xs[1]+18,34,12,'#e1c477');}
function render(){ctx.clearRect(0,0,1120,570);ctx.fillStyle='#090b10';ctx.fillRect(0,0,1120,570);let p=phase,local=elapsed/durations[p];
// Persistent input panel
rounded(18,65,230,285,10,'#111a29','#3c8294');label('INPUT IMAGE',133,48,13,'#e1c477');gridInput(42,91,182,p===1,local);label('28 × 28 grayscale pixels',133,300,12);label(p===1?'3×3 learned kernel scans the input':'784 intensity values',133,320,11,p===1?'#e1c477':'#9da8bc');
// Convolution/filter feature map stage: display actual activation channels.
label('CONV 1',365,48,13,'#e1c477');D.conv1.slice(0,2).forEach((m,i)=>mapTile(m.map,280+i*105,103,80,`Channel ${m.channel}`));label('32 learned filters',440,225,12);label('POOL 1 · 14 × 14 × 32',440,246,11,'#aeb8cb');mapTile(D.pool1,394,265,92,'Mean across 32 channels');
label('CONV 2',635,48,13,'#e1c477');D.conv2.slice(0,2).forEach((m,i)=>mapTile(m.map,530+i*92,103,76,`Channel ${m.channel}`));label('64 learned filters',650,220,12);label('POOL 2 · 7 × 7 × 64',650,241,11,'#aeb8cb');mapTile(D.pool2,610,265,82,'Mean across 64 channels');
// Flow line and moving signal
ctx.setLineDash([5,7]);ctx.strokeStyle='#56627a';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(248,205);ctx.lineTo(270,205);ctx.moveTo(495,205);ctx.lineTo(525,205);ctx.moveTo(805,205);ctx.lineTo(820,205);ctx.stroke();ctx.setLineDash([]);
let pathX=[248,280,394,535,610,710,820,900],px=pathX[Math.min(p,6)]+(pathX[Math.min(p,6)+1]-pathX[Math.min(p,6)])*Math.min(1,local);ctx.beginPath();ctx.arc(px,205,6,0,Math.PI*2);ctx.fillStyle='#e1c477';ctx.shadowColor='#e1c477';ctx.shadowBlur=16;ctx.fill();ctx.shadowBlur=0;
// Dense and output panel
rounded(715,65,385,440,10,'#101521','#5b5360');drawDense(Math.max(0,(p-5)+local));
label('18 representative neurons shown · 128 exist in the model',905,525,10,'#9da8bc');label(`Predicted class: ${D.predicted} · ${(D.probabilities[D.predicted]*100).toFixed(2)}%`,905,550,13,'#e1c477');
// Highlight currently active stage
let regions=[[18,65,230,285],[268,65,238,200],[390,255,100,110],[523,65,290,200],[605,255,100,110],[715,65,385,440],[715,65,385,440],[715,65,385,440]];let q=regions[p];ctx.strokeStyle='#e1c477';ctx.lineWidth=2.5;ctx.strokeRect(q[0]-3,q[1]-3,q[2]+6,q[3]+6);
}
function updateUI(){document.getElementById('stage').textContent=`STEP ${phase+1} / 8 · ${steps[phase]}`;document.getElementById('stat').textContent=phase===7?`Predicted ${D.predicted} · confidence ${(D.probabilities[D.predicted]*100).toFixed(2)}% · all ten scores are model outputs.`:`Tensor flow: ${steps[phase]} · the highlighted stage is active.`}
function advance(dt){elapsed+=dt*speed;while(elapsed>=durations[phase]){elapsed-=durations[phase];if(phase<7)phase++;else{phase=0;elapsed=0;playing=false;document.getElementById('play').textContent='▶ Play';break}}updateUI();render()}
function loop(t){if(!playing)return;if(!last)last=t;let dt=Math.min(.05,(t-last)/1000);last=t;advance(dt);raf=requestAnimationFrame(loop)}
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Ⅱ Pause':'▶ Play';last=0;if(playing)raf=requestAnimationFrame(loop)};
document.getElementById('step').onclick=()=>{playing=false;document.getElementById('play').textContent='▶ Play';phase=Math.min(7,phase+1);elapsed=0;updateUI();render()};
document.getElementById('reset').onclick=()=>{playing=false;phase=0;elapsed=0;last=0;document.getElementById('play').textContent='▶ Play';updateUI();render()};
document.getElementById('speed').oninput=e=>speed=Number(e.target.value);updateUI();render();
</script></body></html>'''.replace("__PAYLOAD__", payload)
