document.getElementById('submit').disabled = true; //Disable at first
document.getElementById('input_lang_text').addEventListener('keyup', e => {
  if (e.target.value == "") {
    document.getElementById('submit').disabled = true;
  }
  else {
    document.getElementById('submit').disabled = false;
  }
});