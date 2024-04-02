import React from "react";
import { useState, useEffect } from "react";

import Card from 'react-bootstrap/Card';
import Row from 'react-bootstrap/Row'
import Col from 'react-bootstrap/Col';
import Button from "react-bootstrap/Button";
import Spinner from "react-bootstrap/Spinner";
import Dropdown from "react-bootstrap/Dropdown";
import DropdownButton from "react-bootstrap/DropdownButton";

import MultiTrackView from './MultiTrackView.react.js'
import SampleMidiDropdown from "../utils/SampleMidiDropdown.js";
import InstListDropdown from "../utils/InstListDropdown.js";
import SaveProjectModal from "./SaveProjectModal.react.js";
import LoadProjectModal from "./LoadProjectModal.react.js";
// import { instrumentMap } from "../utils/InstrumentList";

import { Midi } from '@tonejs/midi'
import { ButtonGroup } from "react-bootstrap";


// Drag & Drop event handler
const handleDragOver = (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
};

// Drag & Drop 한 파일 ArrayBuffer로 저장
const readFileAsArrayBuffer = (file) => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = (event) => {
            resolve(event.target.result);
        };

        reader.onerror = (error) => {
            reject(error);
        };

        reader.readAsArrayBuffer(file);
    });
};


// Main component
const MidiView = (props) => {
    const [midiFile, setMidiFile] = useState();
    const [fileName, setFileName] = useState("Drag and drop MIDI file here")
    const [sampleTitle, setSampleTitle] = useState("Sample MIDI");
    const [instrumentObject, setInstrumentObject] = useState({});
    const [regenTrackIdx, setRegenTrackIdx] = useState(null);
    const [regenInstNum, setRegenInstNum] = useState();
    const [addInstNum, setAddInstNum] = useState(999);
    const [regenTrigger, setRegenTrigger] = useState(0);
    const [isAdding, setIsAdding] = useState(false);
    const [isExtending, setIsExtending] = useState(false);
    const [isInfilling, setIsInfilling] = useState(false);
    const [isDownloading, setIsDownloading] = useState(false);
    const [totalBars, setTotalBars] = useState(4);
    const [barsToRegen, setBarsToRegen] = useState([0, 3]);
    const [currentInstruments, setCurrentInstruments] = useState([]);
    const [infillHighlightBar, setInfillHighlightBar] = useState();
    const [undoSteps, setUndoSteps] = useState([]);
    const [showSaveProjectModal, setShowSaveProjectModal] = useState(false);
    const [showLoadProjectModal, setShowLoadProjectModal] = useState(false);


    // 서버에서 생성해서 반환해준 미디 파일을 멀티트랙 뷰로 넘겨줌
    useEffect(() => {
        if (props.midiBlob) {
            try {
                const newMidiFile = new Midi(props.midiBlob);
                setMidiFile(newMidiFile);
                setUndoSteps([]);
            } catch (error) {
                console.error('Error parsing MIDI file:', error);
            }
        }
    }, [props.midiBlob])


    // 악기 재생성 trigger
    useEffect(() => {
        regenerateSingleInstrument();
    }, [regenTrigger])

    // midiFile 갱신되면 현재 어떤 악기들이 있는지 가져오는 함수
    useEffect(() => {
        if (midiFile) {
            const instrumentsArray = [];
            midiFile.tracks.forEach((track) => {
                track.instrument.percussion ?
                    instrumentsArray.push(-1) :
                    instrumentsArray.push(track.instrument.number);
            })
            // console.log(instrumentsArray);
            setCurrentInstruments(instrumentsArray);
        }
    }, [midiFile])

    // 드래그 앤 드롭으로 올린 미디 파일을 멀티트랙 뷰로 보내고 서버에 전송 가능한 형태로 준비시킴
    const handleFileDrop = async (event) => {
        event.preventDefault();

        const file = event.dataTransfer.files[0];

        if (file) {
            try {
                // setMidiFileRaw(file);

                const arrayBuffer = await readFileAsArrayBuffer(file);
                const midi = new Midi(arrayBuffer)

                setMidiFile(midi);
                setFileName(file.name);
            } catch (error) {
                console.error('Error parsing MIDI file:', error);
            }
        }
    };

    // Load Cache MIDI Button
    const LoadCacheMidiButton = () => {
        return (
            <Dropdown>
                <Dropdown.Toggle
                    variant="outline-secondary"
                    className="float-end"
                    id="dropdown-basic"
                    size="sm"
                    style={{ display: 'flex', alignItems: 'center', borderColor: "#c2c2c2" }}
                >
                    <img src="misc_img/cogwheel.png" width="18px" />
                    {/* Menu */}
                </Dropdown.Toggle>
                <Dropdown.Menu>
                    <Dropdown.Item
                        onClick={() => { setShowSaveProjectModal(true); }}
                    >
                        Save Project
                    </Dropdown.Item>
                    <Dropdown.Item
                        onClick={() => { setShowLoadProjectModal(true); }}
                    >
                        Load Project
                    </Dropdown.Item>
                </Dropdown.Menu>
            </Dropdown >
        )
    }

    // 특정 악기를 Regenerate 하도록 하면, 해당 악기 번호와 해당 악기만 제외한 미디 파일을 서버로 전달
    const regenerateSingleInstrument = () => {
        if (midiFile) {
            try {
                let regenPart;
                let newMidi;
                if (totalBars === 4) {
                    console.log("regenerate default")
                    regenPart = "default";

                    newMidi = midiFile.clone();
                    newMidi.tracks.splice(regenTrackIdx, 1);
                } else if (totalBars === 8) {
                    if (barsToRegen[0] === 0 && barsToRegen[1] === 3) {
                        console.log(barsToRegen);
                        regenPart = "front";
                    } else if (barsToRegen[0] === 4 && barsToRegen[1] === 7) {
                        console.log(barsToRegen);
                        regenPart = "back";
                    }
                    newMidi = midiFile.clone();
                    const removedTrack = newMidi.tracks.splice(regenTrackIdx, 1)[0];
                    newMidi.tracks.push(removedTrack);
                }

                sendMidiToServerLambda({ operateType: "regen", midi: newMidi, instNum: regenInstNum, regenPart: regenPart });
            } catch (error) {
                console.error('Error Regenerating Single Instrument:', error)
            }
        }
    }

    // 현재 MIDI File을 서버에 보내고, 추가 혹은 수정된 미디 파일을 받는 함수
    const sendMidiToServerLambda = ({ operateType, midi, instNum, regenBarIndex, regenPart }) => {

        // Save current MIDI file to cache
        cacheMidiState();

        // Create FormData object
        const midiArray = midi.toArray()
        const base64Data = btoa(String.fromCharCode.apply(null, midiArray));

        // Operate Type에 따라 url 및 body 데이터 및 지정
        let url;
        let bodyData;
        if (operateType === "add" || operateType === "regen") {
            url = "https://hye8o7tt0m.execute-api.ap-northeast-2.amazonaws.com/default/codePlaySendMidiToServer2"; // AWS API Gateway Endpoint
            bodyData = JSON.stringify({
                "midi": base64Data,
                "instnum": instNum,
                "emotion": props.generateConditions.emotion,
                "tempo": props.generateConditions.tempo,
                "genre": props.generateConditions.genre,
                "regenPart": regenPart
            });
            setIsAdding(true);
        } else if (operateType === "extend") {
            // console.log(`Extend Midi to 8 bars`);
            url = "https://eqipz7j6o7.execute-api.ap-northeast-2.amazonaws.com/default/codeplayExtendMidi"; // AWS API Gateway Endpoint
            bodyData = JSON.stringify({
                "midi": base64Data
            })
            setIsExtending(true);
        } else if (operateType === "infill") {
            // console.log(`regenBarIndex: ${regenBarIndex}`);
            url = "https://65yj39pow7.execute-api.ap-northeast-2.amazonaws.com/default/codeplayInfillMidi";
            bodyData = JSON.stringify({
                "midi": base64Data,
                "regenBarIndex": regenBarIndex,
                "totalBars": totalBars,
            })
            setIsInfilling(true);
        } else if (operateType === "audio") {
            url = "https://voi1e5815l.execute-api.ap-northeast-2.amazonaws.com/default/codeplayMidiToAudio";
            bodyData = JSON.stringify({
                "midi": base64Data,
            })
            setIsDownloading(true);
        }

        // Make the POST request using fetch
        fetch(url, {
            method: 'POST',
            headers: { "Content-Type": 'application/json', "Accept": "*/*" },
            body: bodyData
        })
            .then((response) => {
                const reader = response.body.getReader();
                let receivedData = ''; // Variable to store the received data

                // Define a function to recursively read the response body
                function readResponseBody(reader) {
                    return reader.read().then(async ({ done, value }) => {
                        if (done) {
                            console.log('Response body fully received');
                            try {
                                if (operateType !== "audio") {

                                    // Remove the last comma character
                                    receivedData = receivedData.substring(0, receivedData.length - 1);

                                    var numericValues = receivedData.split(',').map(function (item) {
                                        return parseInt(item.trim(), 10); // Convert each item to an integer
                                    });
                                    var uint8Array = new Uint8Array(numericValues);

                                    // Uint8Array 디코딩
                                    const string = new TextDecoder().decode(uint8Array);
                                    let modifiedStr = string.substring(1, string.length - 1);

                                    const dataURI = `data:audio/midi;base64,${modifiedStr}`
                                    const dataURItoBlob = (dataURI) => {

                                        const byteString = atob(dataURI.split(',')[1]);
                                        const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

                                        let ab = new ArrayBuffer(byteString.length);
                                        let ia = new Uint8Array(ab);
                                        for (let i = 0; i < byteString.length; i++) {
                                            ia[i] = byteString.charCodeAt(i);
                                        }

                                        return new Blob([ab], { type: mimeString });
                                    };

                                    const arrayBuffer = await readFileAsArrayBuffer(dataURItoBlob(dataURI));

                                    // operateType에 따라 나눠서 응답 미디 파일 처리
                                    if (operateType === "extend" || operateType === "infill") {

                                        const midi = new Midi(arrayBuffer)

                                        // console.log(instrumentObject);
                                        // console.log(midiFile);

                                        // 1. 기존 MIDI 파일 트랙 순서 저장했다가 다시 덮어씌워주기
                                        const tempTrackInstOrder = {}
                                        Object.entries(midiFile.tracks).forEach(([idx, track]) => {
                                            if (track.instrument.percussion) {
                                                tempTrackInstOrder[-1] = idx;
                                            } else {
                                                tempTrackInstOrder[track.instrument.number] = idx;
                                            }
                                        })

                                        const trackSort = (a, b) => {
                                            if (a.instrument.percussion) {
                                                const orderA = tempTrackInstOrder[-1];
                                                const orderB = tempTrackInstOrder[b.instrument.number];
                                                return orderA - orderB;
                                            } else if (b.instrument.percussion) {
                                                const orderA = tempTrackInstOrder[a.instrument.number];
                                                const orderB = tempTrackInstOrder[-1];
                                                return orderA - orderB;
                                            } else {
                                                const orderA = tempTrackInstOrder[a.instrument.number];
                                                const orderB = tempTrackInstOrder[b.instrument.number];
                                                return orderA - orderB;
                                            }
                                        };
                                        midi.tracks.sort(trackSort);
                                        // console.log(`midi.tracks: ${midi.tracks[0].instrument.number}`)
                                        // console.log(`midi after sort: ${midi}`)

                                        // 2. 기존 MIDI 파일 트랙별 이름 저장했다가 다시 덮어씌워주기
                                        const tempInstNameObject = {}

                                        Object.entries(midiFile.tracks).forEach(([idx, track]) => {
                                            tempInstNameObject[idx] = track.name;
                                        })

                                        midi.tracks.forEach((track, idx) => {
                                            track.name = tempInstNameObject[idx];
                                        })
                                        setMidiFile(midi);

                                    } else if (operateType === "add") {

                                        const midi = new Midi(arrayBuffer)
                                        const lastTrack = midi.tracks[midi.tracks.length - 1];
                                        const newMidi = midiFile.clone();
                                        newMidi.tracks.push(lastTrack);
                                        setMidiFile(newMidi);
                                        setAddInstNum(999);

                                    } else if (operateType === "regen") {

                                        const midi = new Midi(arrayBuffer)
                                        const lastTrack = midi.tracks[midi.tracks.length - 1];
                                        const newMidi = midiFile.clone()
                                        newMidi.tracks[regenTrackIdx] = lastTrack;
                                        setMidiFile(newMidi);
                                        setRegenTrackIdx(null);

                                    }
                                } else if (operateType === "audio") {

                                    // Remove the last comma character
                                    receivedData = receivedData.substring(0, receivedData.length - 1);

                                    // Parse the string and extract the numeric values
                                    var numericValues = receivedData.split(',').map(function (item) {
                                        return parseInt(item.trim(), 10); // Convert each item to an integer
                                    });

                                    // Create a Uint8Array from the numeric values
                                    var uint8Array = new Uint8Array(numericValues);

                                    // const string = new TextDecoder().decode(uint8Array);
                                    const string = new TextDecoder('utf-8').decode(uint8Array);
                                    let modifiedStr = string.substring(1, string.length - 1);
                                    // console.log("Response body fully received");

                                    const dataURI = `data:audio/mpeg;base64,${modifiedStr}`
                                    const dataURItoBlob = (dataURI) => {

                                        const byteString = atob(dataURI.split(',')[1]);
                                        const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

                                        let ab = new ArrayBuffer(byteString.length);
                                        let ia = new Uint8Array(ab);
                                        for (let i = 0; i < byteString.length; i++) {
                                            ia[i] = byteString.charCodeAt(i);
                                        }

                                        return new Blob([ab], { type: mimeString });
                                    };

                                    // console.log(arrayBuffer);
                                    const wavBlob = dataURItoBlob(dataURI)
                                    const blobUrl = URL.createObjectURL(wavBlob);

                                    // Create a download link
                                    const downloadLink = document.createElement('a');
                                    downloadLink.href = blobUrl;
                                    downloadLink.download = `generated_mp3.mp3`;

                                    // Append the link to the document body
                                    document.body.appendChild(downloadLink);

                                    // Trigger the click event on the link
                                    downloadLink.click();

                                    // Remove the link from the document body
                                    document.body.removeChild(downloadLink);

                                    // Revoke the Blob URL to free up resources
                                    URL.revokeObjectURL(blobUrl);
                                    setIsDownloading(false)
                                }
                            } catch (error) {
                                console.error('Error reading file as array buffer:', error);
                            }
                            return;
                        }

                        // Process the received chunk of data (value) here
                        // console.log('Received chunk of data:', value);

                        // receivedData += value;
                        receivedData += value + ',';

                        // Continue reading the next chunk of data
                        return readResponseBody(reader);
                    }).catch((error) => {
                        console.error('Error reading response body:', error);
                    });
                }
                // Start reading the response body
                readResponseBody(reader);
                setIsAdding(false);
                setIsExtending(false);
                setIsInfilling(false);
                setIsDownloading(false);
            })
            .catch(error => {
                props.setShowErrorModal(true);
                props.setErrorLog(error.message);
                setIsAdding(false);
                setIsExtending(false);
                setIsInfilling(false);
                setIsDownloading(false);
            });
    }

    const handleClickAddInst = () => {

        // Server에서 4 bar add인지, 8 bar add인지 구별하게 하는 변수
        let regenPart;
        if (totalBars === 4) {
            regenPart = "default";
        } else if (totalBars === 8) {
            regenPart = "both";
        }

        if (addInstNum >= -1 && addInstNum <= 127) {
            sendMidiToServerLambda({ operateType: "add", midi: midiFile, instNum: addInstNum, regenPart: regenPart });
        } else { // Random Inst Add 하는 경우 예외 처리
            sendMidiToServerLambda({ operateType: "add", midi: midiFile, instNum: 999, regenPart: regenPart });
        }
    }

    const handleClickExtend = () => {
        sendMidiToServerLambda({ operateType: "extend", midi: midiFile });
    }

    const handleClickInfill = (barIndex) => {
        console.log(`bar ${barIndex}'s all tracks will be regenerated`);
        sendMidiToServerLambda({ operateType: "infill", midi: midiFile, regenBarIndex: barIndex });
        setInfillHighlightBar(barIndex);
    }

    const handleDownloadMidi = () => {
        if (midiFile) {
            try {
                const midiArray = midiFile.toArray()
                const midiBlob = new Blob([midiArray])
                // Create a Blob URL for the data
                const blobUrl = URL.createObjectURL(midiBlob);

                // Create a download link
                const downloadLink = document.createElement('a');
                downloadLink.href = blobUrl;
                downloadLink.download = `generated_midi.mid`;

                // Append the link to the document body
                document.body.appendChild(downloadLink);

                // Trigger the click event on the link
                downloadLink.click();

                // Remove the link from the document body
                document.body.removeChild(downloadLink);

                // Revoke the Blob URL to free up resources
                URL.revokeObjectURL(blobUrl);
            } catch (error) {
                console.error('Error downloading MIDI file:', error)
            }
        }
    }

    const handleDownloadAudio = () => {
        sendMidiToServerLambda({ operateType: "audio", midi: midiFile });
    }

    const cacheMidiState = () => {
        setUndoSteps(prevSteps => [...prevSteps, midiFile.clone()]);
    };

    const undoMidiChange = () => {
        if (undoSteps.length > 0) {
            const prevState = undoSteps.pop();
            setMidiFile(prevState);
            // Optionally, you may want to update the UI or trigger MIDI playback with the reverted data
            // For example:
            setUndoSteps([...undoSteps]); // Update cached states
        } else {
            console.log("No more changes to undo.");
        }
    }

    const handleLoadSampleMidi = async (sampleMidiPath) => {
        const midiInstance = await Midi.fromUrl(sampleMidiPath);
        setMidiFile(midiInstance);
    }

    return (
        <>
            <Col xs={props.arrWidth}>
                <Card>
                    <Card.Header as="h5">
                        <Row>
                            <Col className="d-flex align-items-center">
                                Generated Music
                            </Col>
                            <Col>
                                <LoadCacheMidiButton />
                            </Col>
                        </Row>
                    </Card.Header>
                    <Card.Body>
                        {/* <div
                        onDrop={handleFileDrop}
                        onDragOver={handleDragOver}
                        style={{
                            width: '100%',
                            height: '5vh',
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            border: '1.5px dashed #aaa',
                            marginBottom: "10px",
                        }}
                    >
                        <span style={{ color: "gray" }}>
                            <img src="./inst_icons/disc.png" width="25px" className="me-2" />
                            {fileName}
                        </span>
                    </div> */}
                        <MultiTrackView
                            isMobileDevice={props.isMobileDevice}
                            midiFile={midiFile}
                            totalBars={totalBars}
                            isAdding={isAdding}
                            regenTrackIdx={regenTrackIdx}
                            barsToRegen={barsToRegen}
                            isExtending={isExtending}
                            isInfilling={isInfilling}
                            infillHighlightBar={infillHighlightBar}
                            instrumentObject={instrumentObject}
                            isGenerating={props.isGenerating}
                            handleClickInfill={handleClickInfill}
                            setIsInfilling={setIsInfilling}
                            setInfillHighlightBar={setInfillHighlightBar}
                            setTotalBars={setTotalBars}
                            setBarsToRegen={setBarsToRegen}
                            setMidiFile={setMidiFile}
                            setInstrumentObject={setInstrumentObject}
                            setRegenTrackIdx={setRegenTrackIdx}
                            setRegenInstNum={setRegenInstNum}
                            setRegenTrigger={setRegenTrigger}
                        />
                        {midiFile ?
                            <Row className="mt-3">
                                <Col>
                                    <DropdownButton
                                        as={ButtonGroup}
                                        className="float-start"
                                        title={isDownloading ? "Downloading..." : "Download"}
                                        variant="outline-dark"
                                        disabled={isDownloading || props.isGenerating || isAdding || isExtending || isInfilling}
                                    >
                                        <Dropdown.Item
                                            as="button"
                                            key="0"
                                            onClick={handleDownloadMidi}
                                        >
                                            <span>MIDI File (.mid)</span>
                                        </Dropdown.Item>
                                        <Dropdown.Item
                                            as="button"
                                            key="1"
                                            onClick={handleDownloadAudio}
                                        >
                                            <span>Audio File (.mp3)</span>
                                        </Dropdown.Item>
                                    </DropdownButton>
                                    {/* <Button
                                    className="ms-2"
                                    variant="outline-dark"
                                    onClick={handleSaveLocalStorage}
                                >
                                    Save cache
                                </Button> */}
                                    {/* <Button
                                    className="ms-2"
                                    variant="outline-dark"
                                    onClick={handleLoadLocalStorage}
                                    disabled={false}
                                >
                                    Load Cache
                                </Button> */}
                                    <Button
                                        className="ms-2"
                                        variant="outline-dark"
                                        onClick={undoMidiChange}
                                        disabled={!undoSteps.length > 0}
                                    >
                                        Undo
                                    </Button>
                                    {/* <SampleMidiDropdown
                                    sampleTitle={sampleTitle}
                                    handleLoadSampleMidi={handleLoadSampleMidi}
                                    setSampleTitle={setSampleTitle}
                                    isGenerating={props.isGenerating}
                                    isAdding={isAdding}
                                /> */}
                                    {/* <Button
                                    className="float-start ms-2"
                                    variant="danger"
                                    onClick={() => { props.setShowErrorModal(true) }}
                                >
                                    Error!
                                </Button> */}
                                </Col>
                                <Col>
                                    <ButtonGroup className="float-end me-2">
                                        <InstListDropdown
                                            isMobileDevice={props.isMobileDevice}
                                            addInstNum={addInstNum}
                                            currentInstruments={currentInstruments}
                                            setAddInstNum={setAddInstNum}
                                        />
                                        <Button
                                            size={props.isMobileDevice && "sm"}
                                            variant="outline-primary"
                                            onClick={handleClickAddInst}
                                            disabled={props.isGenerating || isAdding || isExtending || isInfilling}
                                        >
                                            {isAdding ? "Adding..." : "Add Inst"}
                                        </Button>
                                    </ButtonGroup>
                                    <Button
                                        disabled={totalBars === 8 || isExtending || isAdding || isInfilling}
                                        size={props.isMobileDevice && "sm"}
                                        variant="outline-dark"
                                        className="float-end me-2"
                                        onClick={handleClickExtend}
                                    >
                                        {isExtending ? "Extending..." : "Extend to 8 bars (+)"}
                                        <Spinner
                                            // size="sm"
                                            className="m-0 p-0"
                                            style={{ width: '0.8rem', height: '0.8rem', borderWidth: '2px', marginLeft: '5px', display: isExtending ? 'inline-block' : 'none' }}
                                            variant="dark"
                                            animation="border"
                                            role="status"
                                        >
                                            <span className="visually-hidden">Loading...</span>
                                        </Spinner>
                                    </Button>
                                </Col>
                            </Row> : null
                        }
                    </Card.Body>
                </Card>
            </Col >
            <SaveProjectModal
                midiFile={midiFile}
                setMidiFile={setMidiFile}
                showSaveProjectModal={showSaveProjectModal}
                setShowSaveProjectModal={setShowSaveProjectModal}
            />
            <LoadProjectModal
                setMidiFile={setMidiFile}
                showSaveProjectModal={showLoadProjectModal}
                setShowLoadProjectModal={setShowLoadProjectModal}
            />
        </>
    )
}

export default MidiView;