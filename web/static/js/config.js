/**
 * --------------------------------------------------------------------------
 * CoreUI Boostrap Admin Template config.js
 * Licensed under MIT (https://github.com/coreui/coreui-free-bootstrap-admin-template/blob/main/LICENSE)
 * --------------------------------------------------------------------------
 */

(() => {
  const THEME = 'coreui-free-bootstrap-admin-template-theme';
  const urlParams = new URLSearchParams(window.location.href.split('?')[1]);
  if (urlParams.get('theme') && ['auto', 'dark', 'light'].includes(urlParams.get('theme'))) {
    localStorage.setItem(THEME, urlParams.get('theme'));
  }
  const email_cbtn = document.getElementById("email_cbtn");
  const setEmailStatus = enabled => {
    const requestBody = {
      es: enabled
    };
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/emailstatus', true);
    xhr.onload = function () {
      if (xhr.status === 200) {
        const checked = JSON.parse(xhr.responseText);
        email_cbtn.checked = checked;
      }
    };
    xhr.send(JSON.stringify(requestBody));
  };
  setEmailStatus(null);
  email_cbtn.addEventListener("click", event => {
    event.preventDefault();
    setEmailStatus(event.target.checked);
  });
})();
//# sourceMappingURL=config.js.map