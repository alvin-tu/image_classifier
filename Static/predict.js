
$("image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop("files")[0];
    reader.readAsDataURL(file);
});

// $("#model-selector").change(function() {
//     loadModel($("#model-selector").val());
// });

let model;
(async function (){
    model = await tf.loadModel(`http://127.0.0.1:81/tfjs-models/MobileNet/model.json`);
}) ();

$("#predict-button").click(async function() {
    let image = $("#selected-image").get(0);
    let tensor = preprocessImage(image);

    let predictions = await model.predict(tensor).data();
    let top5 = Array.from(predictions)
        .map(function (p,i) {
            return {
                probability: p,
                className: IMAGENET_CLASSES[i]
            };
        }).sort(function (a,b){
            return b.probability - a.probability;
        }).slice(0,5);

    $("#prediction-list").empty();
    top5.forEach(function (p) {
        $("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
    });
});

function preprocessImage(image){
    let tensor = tf.fromPixels(image)
        .resizeNearestNeighbor([224,224])
        .toFloat();

    let offset = tf.scalar(127.5);
    return tensor.sub(offset)
        .div(offset)
        .expandDims();    
}

