import React from 'react';

import Modal from 'react-bootstrap/Modal';
import { Container } from 'react-bootstrap';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button'

const InfoModal = (props) => {

    return (
        <Modal
            show={props.showInfoModal}
            onHide={() => props.setShowInfoModal(false)}
            dialogClassName="error-modal"
        >
            <Modal.Header closeButton>
                <Modal.Title>Info</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <Container>
                    <Row>
                        <span>Info Content Coming Soon...</span>
                        <h3>NLP-07 CodePlay 코뿔소 와!!</h3>
                    </Row>
                    <Row className="mt-4 float-end">
                        <Col>
                            <Button
                                variant="secondary"
                                onClick={() => { props.setShowInfoModal(false) }}
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

export default InfoModal;