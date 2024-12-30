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
      canvas.dataset.play = 'true';
      ww.postMessage({
        cmd: 'play'
      });
      canvas.addEventListener('click', e => {
        const c = e.currentTarget;
        const isplay = c.dataset.play === 'true';
        c.dataset.play = isplay ? 'false' : 'true';
        const cmd = isplay ? 'stop' : 'play';
        ww.postMessage({
          cmd: cmd
        });
      });
  }
};
//# sourceMappingURL=index.js.map