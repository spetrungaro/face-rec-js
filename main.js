import * as tf from "@tensorflow/tfjs-node";
import * as faceapi from "face-api.js";
import fs from "fs";
import path from "path";

// Carga el modelo de reconocimiento facial de Face-api
// await faceapi.tf.setBackend("tensorflow");
// await faceapi.tf.enableProdMode();
// await faceapi.tf.ENV.set("DEBUG", false);
// await faceapi.tf.ready();

const MODEL_URL = "./models";
await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL);
await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL);

const TRAIN_FOLDER = "./train";

function image(img) {
    const buffer = fs.readFileSync(img);
    const decoded = tf.node.decodeImage(buffer);
    const casted = decoded.toFloat();
    const result = casted.expandDims(0);
    decoded.dispose();
    casted.dispose();
    return result;
}

async function trainModel() {
    // Lee las subcarpetas en la carpeta "caras"
    const folders = fs.readdirSync(TRAIN_FOLDER);
    // Crea un array de objetos con la ruta de cada foto y el nombre de la carpeta correspondiente
    const labeledDescriptors = folders.map((folderName) => {
        const personPath = path.join(TRAIN_FOLDER, folderName);
        const faceDescriptors = fs
            .readdirSync(personPath)
            .filter(
                (fileName) =>
                    path.extname(fileName) === "jpeg" ||
                    path.extname(fileName) === ".png" ||
                    path.extname(fileName) === ".jpg"
            )
            .map(async (personPicture) => {
                const img = image(path.join(personPath, personPicture));
                const faceTensor = await faceapi
                    .detectSingleFace(img)
                    .withFaceLandmarks()
                    .withFaceDescriptor();
                const faceDescriptor = faceTensor.descriptor;
                console.info(faceDescriptor);
                return faceDescriptor;
            });
        console.info(faceDescriptors);
        // return new faceapi.LabeledFaceDescriptors(folderName, faceDescriptors);
    });
    // Entrena el modelo con los datos de las fotos etiquetadas
    // console.info(await labeledDescriptors);
    // const faceMatcher = new faceapi.FaceMatcher(await labeledDescriptors[0]);

    // console.log(faceMatcher);
    // Serializa el modelo entrenado en un archivo
    // fs.writeFileSync("faceMatcher.json", JSON.stringify(faceMatcher));

    // return faceMatcher;
}

async function recognizeFaces() {
    // Carga el modelo entrenado desde el archivo
    const faceMatcherJson = fs.readFileSync("faceMatcher.json");
    const labeledDescriptors = JSON.parse(faceMatcherJson);
    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors);

    // Lee las fotos que quieres reconocer
    const images = fs
        .readdirSync("./reconocer")
        .filter((fileName) => path.extname(fileName) === ".jpg")
        .map((fileName) => canvas.loadImage(`./reconocer/${fileName}`));

    // Reconoce las caras en las fotos
    const results = await Promise.all(
        images.map(async (img) => {
            const detections = await faceapi
                .detectAllFaces(img)
                .withFaceLandmarks()
                .withFaceDescriptors();
            return detections.map((detection) =>
                faceMatcher.findBestMatch(detection.descriptor)
            );
        })
    );

    // Muestra los resultados
    results.forEach((result, i) => {
        console.log(`Resultados para la foto ${i}:`);
        result.forEach((match, j) => {
            console.log(`Cara ${j}: ${match.label} (${match.distance})`);
        });
    });
}

trainModel();
