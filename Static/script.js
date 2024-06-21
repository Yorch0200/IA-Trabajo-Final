document.getElementById('file').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('preview');
            preview.src = e.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('predict-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const fileInput = document.getElementById('file');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
    });
    const data = await response.json();
    if (data.error) {
        document.getElementById('result').innerText = `Error: ${data.error}`;
    } else {
        document.getElementById('result').innerText = `Predicción: ${data.prediction}`;
    }
});
