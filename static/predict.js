const MODEL_URL =
  "https://raw.githubusercontent.com/shivangidas/image-classifier/master/modelv1/tensorflowjs_model.pb";
const WEIGHTS_URL =
  "https://raw.githubusercontent.com/shivangidas/image-classifier/master/modelv1/weights_manifest.json";
let model;
let IMAGENET_CLASSES = [];
let offset = tf.scalar(128);
async function loadModelAndClasses() {
  $.getJSON(
    "https://raw.githubusercontent.com/shivangidas/image-classifier/master/mobilenet/imagenet_classes.json",
    function(data) {
      $.each(data, function(key, val) {
        IMAGENET_CLASSES.push(val);
      });
    }
  );
  model = await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);
  //console.log("After model is loaded: " + tf.memory().numTensors);
  $(".loadingDiv").hide();
  $("#image-selector").attr("disabled", false);
}
loadModelAndClasses();

$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
        // $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop("files")[0];
    reader.readAsDataURL(file);
});

function readURL(input) {
    if (input.files && input.files[0]) {
      var reader = new FileReader();
  
      reader.onload = function(e) {        
          let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
      };
  
      reader.readAsDataURL(input.files[0]);
  
      reader.onloadend = async function() {
  
        let imageData = document.getElementById("selected-image");

        let pixels1 = tf.fromPixels(imageData);
        let pixel2 = pixels1.resizeNearestNeighbor([224, 224]);
        let pixel3 = pixel2.toFloat();

        let pixels = pixel3.sub(offset);
        let pixels4 = pixels.div(offset);
        let pixels5 = pixels4.expandDims();

        const output = await model.predict(pixels5);
        const predictions = Array.from(output.dataSync())
          .map(function(p, i) {
            return {
              probabilty: p,
              classname: IMAGENET_CLASSES[i]
            };
          })
          .sort((a, b) => b.probabilty - a.probabilty)
          .slice(0, 5);

        var html = "";
        for (let i = 0; i < 5; i++) {
          html += "<li>" + predictions[i].classname + "</li>";
        }
        $(".prediction-list").html(html);
  
        pixels.dispose();
        pixels1.dispose();
        pixel2.dispose();
        pixel3.dispose();
        pixels4.dispose();
        pixels5.dispose();
        output.dispose();
        console.log("After dispose: " + tf.memory().numTensors);
      };
    }
  }