import React from 'react';
import { useState, useEffect } from 'react';

import Modal from 'react-bootstrap/Modal';
import { Container } from 'react-bootstrap';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form'
import Button from 'react-bootstrap/Button'
import Card from 'react-bootstrap/Card'

import '../index.css'

const MAX_PROJECT_NAME_LEN = 30;



const SaveProjectModal = (props) => {
    const [saveMidiName, setSaveMidiName] = useState("My Project");
    const [saveIndex, setSaveIndex] = useState(0);
    const [cachedProjectArray, setCachedProjectArray] = useState(Array.from({ length: 10 }, (_, index) => ({
        key: `empty_${index + 1}`,
        value: "empty"
    })));

    useEffect(() => {
        const newProjectArray = [...cachedProjectArray];
        const savedProjects = getCachedMidis();

        savedProjects.forEach((project, idx) => {
            newProjectArray[parseInt(project.key.split("_")[1]) - 1].key = project.key;
            newProjectArray[parseInt(project.key.split("_")[1]) - 1].value = project.value;
        })

        console.log(newProjectArray)
        newProjectArray.sort((a, b) => {
            const keyA = parseInt(a.key.split("_")[1]);
            const keyB = parseInt(b.key.split("_")[1]);

            if (keyA < keyB) { return -1; }
            if (keyA > keyB) { return 1; }
            return 0;
        });

        setCachedProjectArray(newProjectArray);
        // console.log(newProjectArray)
    }, [props.showSaveProjectModal])

    const getCachedMidis = () => {
        const keys = Object.keys(localStorage);
        const matchingItems = keys.filter(key => key.includes("tempmidi"));
        const items = matchingItems.map(key => {
            return {
                key: key,
                value: localStorage.getItem(key)
            };
        });


        // Set Save Index based on current projects
        if (items.length < 10) {
            setSaveIndex(items.length)
        }
        console.log(`saveIndex: ${saveIndex}`)
        return items;
    }


    const handleSaveToLocalStorage = () => {
        if (props.midiFile) {
            const newMidiFile = props.midiFile.clone();
            newMidiFile.header.name = saveMidiName;
            console.log(newMidiFile);
            const midiJSON = newMidiFile.toJSON();
            localStorage.setItem(`tempmidi_${saveIndex + 1}`, JSON.stringify(midiJSON));
            props.setShowSaveProjectModal(false);
        } else {
            alert("No MIDI file!")
        }
    }

    return (
        <Modal
            show={props.showSaveProjectModal}
            onHide={() => props.setShowSaveProjectModal(false)}
            dialogClassName="save-modal"
        >
            <Modal.Header closeButton>
                <Modal.Title>Save Current Project</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <Container fluid className='m-0 p-0 justify-content-center align-items-center'>
                    <Row>
                        <Col>
                            <Form.Control
                                type="text"
                                placeholder="Enter Project Name"
                                autoFocus
                                // disabled
                                onChange={(event) => {
                                    // console.log(event.target.value.length);
                                    if (event.target.value.length <= MAX_PROJECT_NAME_LEN) {
                                        setSaveMidiName(event.target.value);
                                    } else {
                                        alert(`Project name can't be longer than ${MAX_PROJECT_NAME_LEN} characters!`)
                                    }

                                }}
                            // defaultValue="My Project"
                            // value={sheetsUrl}
                            />
                        </Col>
                    </Row>
                    <Row className='mt-3 mb-2'>
                        <Col>
                            <b>Current Cached Projects (Save to slot : {saveIndex + 1})</b>
                        </Col>
                    </Row>
                    {cachedProjectArray.map((project, idx) => {
                        return (
                            <Row
                                key={idx}
                                style={saveIndex == idx ? { fontWeight: "bold", cursor: "pointer" } : { cursor: "pointer" }}
                                onClick={() => { setSaveIndex(idx) }}
                            >
                                <Col xs={1}>
                                    {idx + 1}
                                </Col>
                                <Col xs={3}>
                                    {project.key !== "empty" ? project.key : "(empty)"}
                                </Col>
                                <Col xs={8}>
                                    {project.value !== "empty" ? JSON.parse(project.value).header.name : "(empty)"}
                                </Col>
                            </Row>
                        )
                    })}
                    <Row className="mt-4 float-end">
                        <Col>
                            <Button
                                className="me-2"
                                variant="primary"
                                onClick={handleSaveToLocalStorage}
                                disabled={false}
                            >
                                Save
                            </Button>
                            <Button
                                variant="secondary"
                                onClick={() => { props.setShowSaveProjectModal(false) }}
                            >
                                Close
                            </Button>
                        </Col>
                    </Row>
                    <Row className="mt-4">
                        <span style={{ color: "#fa5555", fontStyle: "italic" }}>
                            <b> * The projects will be deleted if you clear cache!</b>
                        </span>
                    </Row>
                </Container>
            </Modal.Body>
        </Modal >
    )
}

export default SaveProjectModal;