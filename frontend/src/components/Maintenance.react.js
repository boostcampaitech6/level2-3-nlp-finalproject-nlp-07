import React from 'react';

import Container from 'react-bootstrap/Container';
import Card from 'react-bootstrap/Card'

const Maintenance = () => {
    return (
        <Container fluid className="p-4">
            <Card className="mt-3" style={{ border: "none" }}>
                <Card.Body className="text-center">
                    <img src="./misc_img/maintenance.png" width="200px" style={{ opacity: 0.8 }} />
                    <h4 className="mt-3" style={{ color: "#3b3b3b" }}>Currently under maintenance...</h4>
                    <h4 className="mt-3" style={{ color: "#3b3b3b" }}>We'll be back soon!</h4>
                </Card.Body>
            </Card>
        </Container>
    )
}

export default Maintenance;