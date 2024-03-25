import React from 'react';
import { useState } from 'react';

import NavigationBar from "./NavigationBar.react";
import MidiView from "./MidiView.react";
import TextPromptView from "./TextPromptView.react";
import ErrorModal from "./ErrorModal.react";
import TutorialModal from "./TutorialModal.react";
import InfoModal from "./InfoModal.react";

import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';


const MainView = () => {
    const [isGenerating, setIsGenerating] = useState(false);
    const [midiBlob, setMidiBlob] = useState();
    const [generateConditions, setGenerateConditions] = useState({});
    const [showErrorModal, setShowErrorModal] = useState(false);
    const [showInfoModal, setShowInfoModal] = useState(false);
    const [showTutorialModal, setShowTutorialModal] = useState(false);
    const [errorLog, setErrorLog] = useState("Error");

    const arrWidth = 12
    Object.keys(generateConditions).length && console.log(generateConditions);

    return (
        <>
            <NavigationBar
                setShowTutorialModal={setShowTutorialModal}
                setShowInfoModal={setShowInfoModal}
            />
            <Container fluid className="p-4">
                <Row>
                    <TextPromptView
                        arrWidth={arrWidth}
                        midiBlob={midiBlob}
                        isGenerating={isGenerating}
                        setMidiBlob={setMidiBlob}
                        setGenerateConditions={setGenerateConditions}
                        setShowErrorModal={setShowErrorModal}
                        setErrorLog={setErrorLog}
                        setIsGenerating={setIsGenerating}
                    />
                    <MidiView
                        arrWidth={arrWidth}
                        midiBlob={midiBlob}
                        isGenerating={isGenerating}
                        generateConditions={generateConditions}
                        setMidiBlob={setMidiBlob}
                        setShowErrorModal={setShowErrorModal}
                        setIsGenerating={setIsGenerating}
                        setErrorLog={setErrorLog}
                    />
                </Row>
            </Container>
            <ErrorModal
                errorLog={errorLog}
                showErrorModal={showErrorModal}
                setShowErrorModal={setShowErrorModal}
            />
            <InfoModal
                showInfoModal={showInfoModal}
                setShowInfoModal={setShowInfoModal}
            />
            <TutorialModal
                showTutorialModal={showTutorialModal}
                setShowTutorialModal={setShowTutorialModal}
            />
        </>
    );
}

export default MainView;
