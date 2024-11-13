function toggleDateLogButton(event) {
  if (!event.target.classList.contains('date-log')) {
    return;
  }
  const buttons = event.target.previousElementSibling.querySelectorAll('button');
  buttons.forEach(button => {
    if (button.classList.contains('d-none')) {
      button.classList.remove('d-none');
    } else {
      button.classList.add('d-none');
    }
  });
}
const dateLogs = document.querySelectorAll('.date-log');
dateLogs.forEach(dateLog => {
  dateLog.addEventListener('hidden.coreui.collapse', toggleDateLogButton);
  dateLog.addEventListener('shown.coreui.collapse', toggleDateLogButton);
});
//# sourceMappingURL=events.js.map