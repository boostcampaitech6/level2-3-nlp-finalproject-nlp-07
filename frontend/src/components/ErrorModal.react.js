import React from 'react';

import Modal from 'react-bootstrap/Modal';
import { Container } from 'react-bootstrap';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button'

const ErrorModal = (props) => {

    return (
        <Modal
            show={props.showErrorModal}
            onHide={() => props.setShowErrorModal(false)}
            dialogClassName="error-modal"
        >
            <Modal.Header closeButton>
                <Modal.Title>Oops! An Error Occurred</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <Container>
                    <Row>
                        <span>Error Message : <br />"<b>{props.errorLog}"</b></span>
                    </Row>
                    <Row className="mt-4 float-end">
                        <Col>
                            <Button
                                variant="secondary"
                                onClick={() => { props.setShowErrorModal(false) }}
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

export default ErrorModal;