import * as tf from "@tensorflow/tfjs-node";
import * as faceapi from "@vladmandic/face-api";
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

const TRAIN_DIR = "./train";
const FIND_DIR = "./find";
const MATCHERS_DIR = "./matchers";

function image(img) {
    const buffer = fs.readFileSync(img);
    const decoded = tf.node.decodeImage(buffer);
    const casted = decoded.toFloat();
    const result = casted.expandDims(0);
    decoded.dispose();
    casted.dispose();
    return result;
}

async function trainModel(tenant) {
    // Lee las subcarpetas en la carpeta de "./train/{tenant}"
    let tenantFolder = path.join(TRAIN_DIR, tenant);
    const folders = fs.readdirSync(tenantFolder);

    // Crea un array de objetos con la ruta de cada foto y el nombre de la carpeta correspondiente
    const labeledDescriptors = await Promise.all(
        folders.map(async (folderName) => {
            console.log(folderName);
            const personPath = path.join(tenantFolder, folderName);
            const faceDescriptors = await Promise.all(
                fs
                    .readdirSync(personPath)
                    .filter(
                        (fileName) =>
                            path.extname(fileName) === ".jpeg" ||
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
                    })
            );
            console.info(faceDescriptors);
            return new faceapi.LabeledFaceDescriptors(
                folderName,
                faceDescriptors
            );
        })
    );
    // Entrena el modelo con los datos de las fotos etiquetadas
    console.info(labeledDescriptors);
    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors);

    // Serializa el modelo entrenado en un archivo
    let tenantFile = path.join(MATCHERS_DIR, `${tenant}.json`);
    fs.writeFileSync(tenantFile, JSON.stringify(faceMatcher));

    // Por si lo usamos de manera secuencial despues
    return faceMatcher;
}

async function recognizeFaces(tenant) {
    // Carga el modelo entrenado desde el archivo
    let tenantFile = path.join(MATCHERS_DIR, `${tenant}.json`);
    const faceMatcherJson = fs.readFileSync(tenantFile);
    const faceMatcherParsed = JSON.parse(faceMatcherJson);
    const faceMatcher = faceapi.FaceMatcher.fromJSON(faceMatcherParsed);

    // // Lee las fotos que quieres reconocer
    const images = fs
        .readdirSync(FIND_DIR)
        .filter(
            (fileName) =>
                path.extname(fileName) === ".jpeg" ||
                path.extname(fileName) === ".png" ||
                path.extname(fileName) === ".jpg"
        )
        .map((fileName) => image(path.join(FIND_DIR, fileName)));

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

const trained = await trainModel("uejn");
//recognizeFaces("uejn");
