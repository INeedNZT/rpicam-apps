const canvas = document.createElement('canvas');
document.body.getElementsByClassName('canvas-container')[0].append(canvas);

// Create h264 player
const uri = `ws://${document.location.host}/live`;
const ww = new Worker('/js/http-live-player-worker.js');
const ofc = canvas.transferControlToOffscreen();
ww.postMessage({
  cmd: 'init',
  canvas: ofc
}, [ofc]);
ww.postMessage({
  cmd: 'connect',
  url: uri
});
ww.onmessage = e => {
  const msg = e.data;
  switch (msg.cmd) {
    case 'canvasReady':
      canvas.style.width = `${msg.width}px`;
      canvas.style.height = `${msg.height}px`;
      ww.postMessage({
        cmd: 'play'
      });
  }
};
//# sourceMappingURL=index.js.map