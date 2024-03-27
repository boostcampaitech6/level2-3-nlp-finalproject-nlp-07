import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import NavDropdown from 'react-bootstrap/NavDropdown';

function NavBar(props) {
    return (
        <Navbar style={{ backgroundColor: "#A181FD" }} data-bs-theme="dark">
            <Container className="p-1 ms-4 me-3" fluid>
                <a href="/">
                    <img
                        src="./../../logo192.png"
                        width="50"
                        height="50"
                        className="d-inline-block align-top"
                        alt="Codeplay logo"
                    />
                </a>
                <Navbar.Brand style={{ fontSize: "25px" }} href="/" className="ms-3">
                    CodePlay!
                </Navbar.Brand>
                <Navbar.Toggle aria-controls="basic-navbar-nav" />
                <Navbar.Collapse className="justify-content-end">
                    <Nav className="me-auto">
                        <Nav.Link
                            onClick={() => { props.setShowInfoModal(true) }}
                        >
                            Info
                        </Nav.Link>
                        <Nav.Link
                            onClick={() => { props.setShowTutorialModal(true) }}
                        // href="#link"
                        >
                            Tutorial
                        </Nav.Link>
                        {/* <NavDropdown title="Menu" id="basic-nav-dropdown">
                            <NavDropdown.Item href="#action/3.1">Action</NavDropdown.Item>
                            <NavDropdown.Item href="#action/3.2">
                                Another action
                            </NavDropdown.Item>
                            <NavDropdown.Item href="#action/3.3">Something</NavDropdown.Item>
                            <NavDropdown.Divider />
                            <NavDropdown.Item href="#action/3.4">
                                Separated link
                            </NavDropdown.Item>
                        </NavDropdown> */}
                    </Nav>
                    <Navbar.Text
                        style={{ textAlign: "right", fontSize: "0.6em" }}
                        className="m-0 p-0"
                    >
                        <span>codeplay.nlp@gmail.com</span>
                        <br />
                        <span>© 2024 NLP07-CodePlay CO. ALL RIGHTS RESERVED</span>
                    </Navbar.Text>
                </Navbar.Collapse>

            </Container>
        </Navbar>
    );
}

export default NavBar;