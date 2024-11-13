const player = videojs('#my-video', {
  playbackRates: [0.25, 0.5, 1, 2, 4]
});
player.fill(true);
player.playlist([{
  name: 'First Video',
  thumbnail: 'http://media.w3.org/2010/05/sintel/poster.png',
  sources: [{
    src: 'http://media.w3.org/2010/05/sintel/trailer.mp4',
    type: 'video/mp4'
  }],
  poster: 'http://media.w3.org/2010/05/sintel/poster.png'
}, {
  name: 'Second Video',
  thumbnail: 'http://media.w3.org/2010/05/bunny/poster.png',
  sources: [{
    src: 'http://media.w3.org/2010/05/bunny/trailer.mp4',
    type: 'video/mp4'
  }],
  poster: 'http://media.w3.org/2010/05/bunny/poster.png'
}, {
  name: 'Third Video',
  thumbnail: 'http://vjs.zencdn.net/v/oceans.png',
  sources: [{
    src: 'http://vjs.zencdn.net/v/oceans.mp4',
    type: 'video/mp4'
  }],
  poster: 'http://vjs.zencdn.net/v/oceans.png'
}, {
  name: 'Fourth Video',
  thumbnail: 'http://media.w3.org/2010/05/bunny/poster.png',
  sources: [{
    src: 'http://media.w3.org/2010/05/bunny/movie.mp4',
    type: 'video/mp4'
  }],
  poster: 'http://media.w3.org/2010/05/bunny/poster.png'
}, {
  name: 'Fifth Video',
  thumbnail: 'http://media.w3.org/2010/05/video/poster.png',
  sources: [{
    src: 'http://media.w3.org/2010/05/video/movie_300.mp4',
    type: 'video/mp4'
  }],
  poster: 'http://media.w3.org/2010/05/video/poster.png'
}]);

// Play through the playlist automatically.
player.playlist.autoadvance(0);
player.playlistUi({
  horizontal: true
});
player.ready(() => {
  const images = document.querySelectorAll('.vjs-playlist-thumbnail img');
  images.forEach(function (img) {
    img.addEventListener('dragstart', function (event) {
      event.preventDefault();
    });
    img.setAttribute('data-src', img.getAttribute('src'));
    img.setAttribute('src', '/assets/img/placeholder-video.png');
    img.classList.add('lozad');
    img.removeAttribute('loading');
    const observer = lozad();
    observer.observe();
  });
});
//# sourceMappingURL=playback.js.map