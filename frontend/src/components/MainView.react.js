import React from 'react';
import { useState, useEffect } from 'react';

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
    const [generationCount, setGenerationCount] = useState(null);

    // Detect if user accessed via Mobile and logs
    const isMobileDevice = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    // console.log(isMobileDevice === true ? "Mobile Environment" : "PC Environment");

    // Assign TextPrompt & MidiView Width
    const arrWidth = 12

    // Log Generation Condition returned from Model 1
    // Object.keys(generateConditions).length && console.log(generateConditions);

    // 총 생성 수 가져오는 함수
    const getGenerationCount = () => {
        
        let url;
        url = "https://78d4shtwvg.execute-api.ap-northeast-2.amazonaws.com/default/codeplayGetGenerationCount";
        
        fetch(
            url,
            {
                method: 'GET',
                headers: {
                    "Content-Type": "application/json",
                    // "Accept": "*/*"
                },
            }
        )
            .then((response) => {
                const reader = response.body.getReader();
                let receivedData = ''; // Variable to store the received data
        
                // Define a function to recursively read the response body
                function readResponseBody(reader) {
                  return reader.read().then(async ({ done, value }) => {
                    if (done) {
                      // console.log('Received data:', receivedData); // Access the received data here
                      try {
                        // console.log('Response body fully received');
                      } catch (error) {
                        console.error('Error reading file as array buffer:', error);
                      }
                      return;
                    }
        
                    // Uint8Array 디코딩
                    const string = new TextDecoder().decode(value);
                    const responseJson = JSON.parse(string);

                    // console.log(responseJson);
                    setGenerationCount(responseJson.total_generate);
                    receivedData += value;
        
                    // Continue reading the next chunk of data
                    return readResponseBody(reader);
                  }).catch((error) => {
                    console.error('Error reading response body:', error);
                  });
                }
        
                // Start reading the response body
                readResponseBody(reader);
        
            })
            .catch((error) => {
                console.error(error);
            });
    }

    useEffect(() => {
        getGenerationCount();
    }, [isGenerating])

    return (
        <>
            <NavigationBar
                isMobileDevice={isMobileDevice}
                generationCount={generationCount}
                setShowTutorialModal={setShowTutorialModal}
                setShowInfoModal={setShowInfoModal}
            />
            <Container fluid className="p-4">
                <Row>
                    <TextPromptView
                        isMobileDevice={isMobileDevice}
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
                        isMobileDevice={isMobileDevice}
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
