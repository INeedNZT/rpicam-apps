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
  const requestEventLogs = event => {
    const logContainer = event.target.querySelector('.log-container');
    const event_id = event.target.parentNode.dataset.id;
    const requestBody = {
      ei: event_id
    };
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/eventlogs', true);
    xhr.onload = function () {
      if (xhr.status === 200) {
        const data = JSON.parse(xhr.responseText);
        const sorted_data = data.sort((a, b) => Number(a.log_time) - Number(b.log_time));
        const parentId = event.target.id;
        const logTemplate = event.target.querySelector('.log');
        for (let index = 0; index < sorted_data.length; index++) {
          const item = sorted_data[index];
          const newLog = logTemplate.cloneNode(true);
          newLog.querySelector('.log-name').textContent = getLogName(item.type);
          newLog.querySelector('.log-time').textContent = item.log_time_str;
          const typeElement = newLog.querySelector('.log-type');
          typeElement.textContent = getLogType(item.type);
          typeElement.classList.add(getLogClass(item.type));
          newLog.querySelector('img').setAttribute('src', item.snapshot_path);
          newLog.style.display = 'block';
          const element = newLog.querySelector('#log');
          element.id = parentId + '_log_' + index;
          const buttons = newLog.querySelectorAll('[data-coreui-target]');
          buttons.forEach(button => {
            const targetId = button.getAttribute('data-coreui-target');
            if (targetId && targetId.startsWith('#log')) {
              button.setAttribute('data-coreui-target', '#' + parentId + '_log_' + index);
            }
          });
          event.target.querySelector('.loading').style.display = 'none';
          logContainer.append(newLog);
        }
      } else {
        errorAlert(`Request failed with status ${xhr.status}`);
      }
    };
    setTimeout(() => {
      xhr.send(JSON.stringify(requestBody));
    }, 500);
  };
  const toggleDateLogButton = event => {
    if (!event.target.classList.contains('date-log')) {
      return;
    }
    for (const button of event.target.previousElementSibling.querySelectorAll('button')) {
      if (button.classList.contains('d-none')) {
        button.classList.remove('d-none');
      } else {
        button.classList.add('d-none');
      }
    }
    if (event.type == 'shown.coreui.collapse') {
      // clear old logs
      event.target.querySelectorAll('.log').forEach(logElement => {
        if (logElement.style.display != 'none') {
          logElement.remove();
        }
      });
      event.target.querySelector('.loading').style.display = 'block';
      requestEventLogs(event);
    }
  };
  const container = document.getElementById('card-container');
  container.addEventListener('hidden.coreui.collapse', event => {
    if (event.target.matches('.date-log')) {
      toggleDateLogButton(event);
    }
  });
  container.addEventListener('shown.coreui.collapse', event => {
    if (event.target.matches('.date-log')) {
      toggleDateLogButton(event);
    }
  });
  const getLogName = type => {
    switch (type) {
      case '0':
        return 'None';
      case '1':
        return 'Motion Detected';
      case '2':
        return 'Face Detected';
    }
  };
  const getLogType = type => {
    switch (type) {
      case '0':
        return 'None';
      case '1':
        return 'Low Risk';
      case '2':
        return 'High Risk';
    }
  };
  const getLogClass = type => {
    switch (type) {
      case '0':
        return '';
      case '1':
        return 'text-bg-warning';
      case '2':
        return 'text-bg-danger';
    }
  };
  window.addEventListener('load', () => {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', '/api/survevents', true);
    xhr.onload = function () {
      if (xhr.status === 200) {
        const data = JSON.parse(xhr.responseText);
        const sorted_data = data.sort((a, b) => Number(b.start_time) - Number(a.start_time));
        const cardTemplate = document.querySelector('.card');
        for (let index = 0; index < sorted_data.length; index++) {
          const item = sorted_data[index];
          const newCard = cardTemplate.cloneNode(true);
          newCard.dataset.id = item.event_id;
          newCard.querySelector('.event-time').innerHTML = item.start_date_str + "&nbsp;&nbsp;" + item.start_time_str;
          const typeElement = newCard.querySelector('.event-type');
          typeElement.textContent = getLogName(item.type);
          typeElement.classList.add(getLogClass(item.type));
          newCard.style.display = 'block';
          const element = newCard.querySelector('#event');
          element.id = 'event_' + index;
          const buttons = newCard.querySelectorAll('[data-coreui-target]');
          buttons.forEach(button => {
            const targetId = button.getAttribute('data-coreui-target');
            if (targetId && targetId.startsWith('#event')) {
              button.setAttribute('data-coreui-target', '#event_' + index);
            }
          });
          container.append(newCard);
        }
      } else {
        errorAlert(`Request failed with status ${xhr.status}`);
      }
    };
    xhr.send();
  });
})();
//# sourceMappingURL=event.js.map