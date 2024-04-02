import React from 'react';
import { useState, useEffect } from 'react';

import Modal from 'react-bootstrap/Modal';
import { Container } from 'react-bootstrap';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form'
import Button from 'react-bootstrap/Button'
import Card from 'react-bootstrap/Card'

import { Midi } from '@tonejs/midi'

import '../index.css'




const LoadProjectModal = (props) => {
    const [loadIndex, setLoadIndex] = useState(0);
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
        items.sort((a, b) => {
            const keyA = parseInt(a.key.split("_")[1]);
            const keyB = parseInt(b.key.split("_")[1]);

            if (keyA < keyB) { return -1; }
            if (keyA > keyB) { return 1; }
            return 0;
        });

        // Set Save Index based on current projects
        console.log(`loadIndex: ${loadIndex}`)
        return items;
    }


    const handleLoadFromLocalStorage = () => {
        const midiJSONString = localStorage.getItem(`tempmidi_${loadIndex + 1}`);
        if (midiJSONString) {
            const midiJSON = JSON.parse(midiJSONString);
            const midiData = new Midi();

            midiData.fromJSON(midiJSON);
            props.setMidiFile(midiData);
            props.setShowLoadProjectModal(false);
        } else {
            alert("File is empty")
        }
    }

    return (
        <Modal
            show={props.showSaveProjectModal}
            onHide={() => props.setShowLoadProjectModal(false)}
            dialogClassName="load-modal"
        >
            <Modal.Header closeButton>
                <Modal.Title>Load Project</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <Container fluid className='m-0 p-0 justify-content-center align-items-center'>
                    {cachedProjectArray.map((project, idx) => {
                        return (
                            <Row
                                key={idx}
                                style={loadIndex == idx ? { fontWeight: "bold", cursor: "pointer" } : { cursor: "pointer" }}
                                onClick={() => { setLoadIndex(idx) }}
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
                                onClick={handleLoadFromLocalStorage}
                                disabled={false}
                            >
                                Load
                            </Button>
                            <Button
                                variant="secondary"
                                onClick={() => { props.setShowLoadProjectModal(false) }}
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

export default LoadProjectModal;