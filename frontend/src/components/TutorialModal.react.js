import React from 'react';

import Modal from 'react-bootstrap/Modal';
import { Container } from 'react-bootstrap';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Card from 'react-bootstrap/Card'
import Button from 'react-bootstrap/Button'

const TutorialModal = (props) => {

    return (
        <Modal
            show={props.showTutorialModal}
            onHide={() => props.setShowTutorialModal(false)}
            dialogClassName="tutorial-modal"
        >
            <Modal.Header closeButton>
                <Modal.Title>Tutorial</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <Container className='m-0 p-0'>
                    <Card style={{ border: "2px solid #618eff" }}>
                        <Card.Header as="h3" style={{ color: "navy", backgroundColor: "#bdcfff" }}>
                            ① Generate Music
                        </Card.Header>
                        <Card.Body>
                            <Row>
                                <img src="./misc_img/tutorial1.png" width="1500px" style={{ objectFit: "cover", objectPosition: "center top", height: "220px" }} />
                                {/* <img src="./misc_img/tutorial1.png" width="1500px" /> */}
                            </Row>
                        </Card.Body>
                    </Card>
                    <Card className='mt-3' style={{ border: "2px solid #618eff" }}>
                        <Card.Header as="h3" style={{ color: "navy", backgroundColor: "#bdcfff" }}>
                            ② Edit Music
                        </Card.Header>
                        <Card.Body>
                            <Row>
                                <img src="./misc_img/tutorial2.png" width="1500px" />
                            </Row>
                        </Card.Body>
                    </Card>
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