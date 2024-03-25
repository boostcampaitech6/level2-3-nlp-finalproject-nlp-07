import React from 'react';

import Modal from 'react-bootstrap/Modal';
import { Container } from 'react-bootstrap';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button'

const TutorialModal = (props) => {

    return (
        <Modal
            show={props.showTutorialModal}
            onHide={() => props.setShowTutorialModal(false)}
            dialogClassName="error-modal"
        >
            <Modal.Header closeButton>
                <Modal.Title>Tutorial</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <Container>
                    <Row>
                        <span>Tutorial Content Coming Soon...</span>
                    </Row>
                    <Row className="mt-4 float-end">
                        <Col>
                            <Button
                                variant="secondary"
                                onClick={() => { props.setShowTutorialModal(false) }}
                            >
                                Close
                            </Button>
                        </Col>
                    </Row>
                </Container>
            </Modal.Body>
        </Modal >
    )
}

export default TutorialModal;