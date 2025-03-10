// You can add interactive JS features, like form validation or image preview

// Show a preview of the uploaded image before submitting
document.querySelector('input[type="file"]').addEventListener('change', function(event) {
    var file = event.target.files[0];
    if (file) {
        var reader = new FileReader();
        reader.onload = function(e) {
            var imgPreview = document.createElement('img');
            imgPreview.src = e.target.result;
            imgPreview.width = 300; // You can set the desired width
            var predictionResultDiv = document.querySelector('.prediction-result');
            predictionResultDiv.innerHTML = ''; // Clear previous result
            predictionResultDiv.appendChild(imgPreview);
        };
        reader.readAsDataURL(file);
    }
});
