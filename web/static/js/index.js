const canvas = document.createElement('canvas');
document.body.getElementsByClassName('canvas-container')[0].appendChild(canvas);

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
ww.postMessage({
  cmd: 'play'
});

// Expose instance for button callbacks
// window.wsavc = {
//   playStream() {
//     ww.postMessage({
//       cmd: 'play'
//     })
//   },
//   stopStream() {
//     ww.postMessage({
//       cmd: 'stop'
//     })
//   },
//   disconnect() {
//     ww.postMessage({
//       cmd: 'disconnect'
//     })
//   }
// }
//# sourceMappingURL=index.js.map