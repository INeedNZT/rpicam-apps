(() => {
  const errorAlert = msg => {
    const alertPlaceholder = document.getElementById('alert-container');
    const appendAlert = (message, type) => {
      const wrapper = document.createElement('div');
      wrapper.innerHTML = [`<div class="alert alert-${type} alert-dismissible fade" role="alert">`, `   <div>${message}</div>`, '</div>'].join('');
      const al = new coreui.Alert(wrapper);
      alertPlaceholder.append(wrapper);
      setTimeout(() => {
        wrapper.firstElementChild.classList.add('show');
      }, 50);
      setTimeout(() => {
        al.close();
      }, 5000);
    };
    appendAlert(msg, 'danger');
  };
  const query = window.location.search.slice(1);
  const [key, value] = query.split('=');
  const requestBody = {};
  requestBody[key] = Number(value);
  const playlist = [];
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/playlist', false);
  xhr.onload = function () {
    if (xhr.status === 200) {
      const data = JSON.parse(xhr.responseText);
      for (const item of data) {
        playlist.push({
          name: item.start_time_str,
          start: item.start_time,
          thumbnail: item.thumb_path,
          sources: [{
            src: item.m3u8_path,
            type: 'application/x-mpegURL'
          }],
          poster: item.thumb_path
        });
      }
      playlist.sort((a, b) => {
        return a.start - b.start;
      });
    } else {
      errorAlert(`Request failed with status ${xhr.status}`);
    }
  };
  xhr.onerror = function (e) {
    errorAlert(`Error Status: ${e.target.status}`);
  };
  xhr.send(JSON.stringify(requestBody));
  const player = videojs('#surv-video', {
    playbackRates: [0.25, 0.5, 1, 2, 4]
  });
  player.fill(true);
  player.playlist(playlist);

  // Play through the playlist automatically.
  player.playlist.autoadvance(0);
  player.playlistUi({
    horizontal: true
  });
  player.ready(() => {
    for (const img of document.querySelectorAll('.vjs-playlist-thumbnail img')) {
      img.addEventListener('dragstart', event => {
        event.preventDefault();
      });
      img.setAttribute('data-src', img.getAttribute('src'));
      img.setAttribute('src', '/assets/img/placeholder-video.png');
      img.classList.add('lozad');
      img.removeAttribute('loading');
      const observer = lozad();
      observer.observe();
    }
  });
})();
//# sourceMappingURL=hls.js.map