(() => {
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
  };
  for (const dateLog of document.querySelectorAll('.date-log')) {
    dateLog.addEventListener('hidden.coreui.collapse', toggleDateLogButton);
    dateLog.addEventListener('shown.coreui.collapse', toggleDateLogButton);
  }
})();
//# sourceMappingURL=events.js.map