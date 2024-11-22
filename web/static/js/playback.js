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
  window.addEventListener('load', () => {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', '/api/survfootage', true);
    xhr.onload = function () {
      if (xhr.status === 200) {
        const data = JSON.parse(xhr.responseText);
        const sorted_data = data.sort((a, b) => Number(b.date) - Number(a.date));
        const cardTemplate = document.querySelector('.surv-card');
        const container = document.getElementById('surv-card-container');
        const loadingPlaceholder = document.getElementById('loading-placeholder');
        for (const item of sorted_data) {
          const newCard = cardTemplate.cloneNode(true);
          newCard.querySelector('.card-title').textContent = item.date_str;
          newCard.querySelector('.card-img-top').setAttribute('src', item.thumb_path);
          newCard.querySelector('a').href = `hls.html?st=${item.date}`;
          newCard.style.display = 'block';
          container.append(newCard);
        }
        loadingPlaceholder.style.display = 'none';
      } else {
        errorAlert(`Request failed with status ${xhr.status}`);
      }
    };

    // Smooth the animation
    setTimeout(() => {
      xhr.send();
    }, 500);
  });
})();
//# sourceMappingURL=playback.js.map