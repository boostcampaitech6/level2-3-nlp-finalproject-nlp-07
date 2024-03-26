import React from 'react';
import { useState, useEffect } from "react";

import Card from 'react-bootstrap/Card';
import Form from 'react-bootstrap/Form';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import InputGroup from 'react-bootstrap/InputGroup';
import Button from 'react-bootstrap/Button';
import Dropdown from "react-bootstrap/Dropdown";
import Spinner from 'react-bootstrap/Spinner';

import { examplePrompts } from "../utils/examplePrompts.js";
import '../index.css'

const MAX_PROMPT_LEN = 200;

// Server에서 받은 파일 ArrayBuffer로 저장
const readFileAsArrayBuffer = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (event) => {
      resolve(event.target.result);
    };

    reader.onerror = (error) => {
      reject(error);
    };
    reader.readAsArrayBuffer(file);
  });
};

// Example Prompts 중 n개 뽑아오는 함수
const returnRandomPrompts = (n, lang) => {

  let exampleObjByLanguage;
  if (lang == "EN") {
    exampleObjByLanguage = examplePrompts["EN"];
  } else if (lang == "KR") {
    exampleObjByLanguage = examplePrompts["KR"];
  }

  const promptsArray = Object.entries(exampleObjByLanguage);
  const result = [];
  const len = promptsArray.length;

  if (n >= len) {
    return promptsArray;
  }

  const indices = new Set();

  while (indices.size < n) {
    indices.add(Math.floor(Math.random() * len));
  }

  for (let index of indices) {
    result.push(promptsArray[index]);
  }

  return result;
}


const TextPromptView = (props) => {
  const [language, setLanguage] = useState("EN")
  const [prompt, setPrompt] = useState("");
  const [showTextPrompt, setShowTextPrompt] = useState(true);
  const [examplePromptsObj, setExamplePromptsObj] = useState(returnRandomPrompts(3, language));
  // const [examplePromptsObj, setExamplePromptsObj] = useState({});

  useEffect(() => {
    const newPromptsObj = returnRandomPrompts(3, language);
    setExamplePromptsObj(newPromptsObj);
  }, [language])

  const LangSelectButton = () => {
    return (
      <Dropdown>
        <Dropdown.Toggle
          variant="outline-secondary"
          className="float-end me-2"
          id="dropdown-basic"
          size="sm"
        >
          {language}
        </Dropdown.Toggle>
        <Dropdown.Menu>
          <Dropdown.Item
            onClick={() => { setLanguage("EN") }}
          >
            English
          </Dropdown.Item>
          <Dropdown.Item
            onClick={() => { setLanguage("KR") }}
          >
            Korean
          </Dropdown.Item>
        </Dropdown.Menu>
      </Dropdown >
    )
  }

  const PromptCards = ({ prompt }) => {
    const [isHovered, setIsHovered] = useState(false);

    const handleClickExample = () => {
      setPrompt(prompt);
      handleClickGenerate();
    }

    const buttonStyle = {
      backgroundColor: isHovered ? '#f0f0f0' : 'white',
      borderColor: "#dbdbdb",
      color: "#7a7a7a",
      cursor: 'pointer',
      // width: '23rem'
    };

    return (
      <Button
        disabled={props.isGenerating}
        style={buttonStyle}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        onClick={handleClickExample}
      >
        {prompt}
      </Button>
    );
  }

  const PromptRefresh = () => {
    const [isHovered, setIsHovered] = useState(false);

    const buttonStyle = {
      backgroundColor: isHovered ? '#f0f0f0' : 'white',
      borderColor: "#dbdbdb",
      color: "#7a7a7a",
      cursor: 'pointer',
    };

    return (
      <Button
        size="sm"
        disabled={props.isGenerating}
        style={buttonStyle}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        onClick={() => { setExamplePromptsObj(returnRandomPrompts(3, language)) }}
      >
        ↺
      </Button>
    );
  }



  const sendLambdaRequest = () => {
    props.setIsGenerating(true);
    fetch(
      "https://zab3ww1o85.execute-api.ap-northeast-2.amazonaws.com/default/codeplayGenerateFromPrompt",
      {
        method: 'POST',
        headers: {
          "Content-Type": "application/json",
          "Accept": "*/*"
        },
        body: JSON.stringify({
          "prompt": prompt,
        }),
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
                console.log('Response body fully received');
              } catch (error) {
                console.error('Error reading file as array buffer:', error);
              }
              return;
            }

            // Uint8Array 디코딩

            const string = new TextDecoder().decode(value);
            const responseJson = JSON.parse(string);
            const conditions = responseJson.condition;
            const [emotion, tempo, genre] = [conditions[0], conditions[1], conditions[2]]
            props.setGenerateConditions((prev) => {
              return { ...prev, ['emotion']: emotion, ['tempo']: tempo, ['genre']: genre };
            });
            const fileContent = responseJson.file_content;

            // Blob 생성
            const dataURI = `data:audio/midi;base64,${fileContent}`
            const dataURItoBlob = (dataURI) => {

              const byteString = atob(dataURI.split(',')[1]);
              const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

              let ab = new ArrayBuffer(byteString.length);
              let ia = new Uint8Array(ab);
              for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
              }

              return new Blob([ab], { type: mimeString });
            };

            const arrayBuffer = await readFileAsArrayBuffer(dataURItoBlob(dataURI));
            props.setMidiBlob(arrayBuffer);

            receivedData += value;

            // Continue reading the next chunk of data
            return readResponseBody(reader);
          }).catch((error) => {
            console.error('Error reading response body:', error);
          });
        }

        // Start reading the response body
        readResponseBody(reader);

        // Set Generating to false when done generating
        props.setIsGenerating(false)
      })
      .catch((error) => {
        console.error(error);
        props.setShowErrorModal(true);
        props.setErrorLog(error.message);
        props.setIsGenerating(false);
      });

  };

  const handleClickGenerate = () => {
    if (props.midiBlob) {
      if (window.confirm(`Delete current tracks and generate new tracks?`)) {
        props.setGenerateConditions({})
        sendLambdaRequest();
      } else {
        return;
      }
    } else {
      props.setGenerateConditions({})
      sendLambdaRequest();
    }
  }

  return (
    <Col
      xs={12 - props.arrWidth}
      className='mb-2'
    >
      <Card>
        <Card.Header as="h5">
          <Row>
            <Col className="d-flex align-items-center">
              Text Prompt
            </Col>
            <Col>
              <Button
                className="float-end"
                variant="outline-secondary"
                size="sm"
                onClick={() => { setShowTextPrompt((prev) => !prev) }}
              >
                {showTextPrompt ? "Hide" : "Show"}
              </Button>
              <LangSelectButton />
            </Col>
          </Row>
        </Card.Header>
        {showTextPrompt ?
          <Card.Body>
            <Row>
              <InputGroup>
                <Form.Control
                  id="prompt-input-field"
                  type="text"
                  placeholder={language === "EN" ? "Enter your prompt..." : language === "KR" ? "어떤 음악을 만들고 싶으신가요?" : null}
                  value={prompt}
                  onChange={(event) => {
                    console.log(event.target.value.length);
                    if (event.target.value.length <= MAX_PROMPT_LEN) {
                      setPrompt(event.target.value);
                    } else {
                      alert(`Prompts can't be longer than ${MAX_PROMPT_LEN} characters!`)
                    }

                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      // handleClickGenerate()
                    } else if (event.key === "Escape") {
                      setPrompt("")
                    }
                  }}
                  autoFocus="autofocus"
                />
              </InputGroup>
            </Row>
            <Row className="mt-4">
              <Col xs={10}>
                <Row>
                  {examplePromptsObj.map((example) => {
                    return (
                      <Col key={example[0]} md={4} className="d-flex justify-content-center align-items-center">
                        <PromptCards prompt={example[1]} />
                      </Col>
                    )
                  })}
                </Row>
              </Col>
              <Col xs={1} className="p-0" style={{ display: 'flex', alignItems: 'center' }}>
                <div style={{ marginRight: 'auto' }}>
                  <PromptRefresh />
                </div>

              </Col>
              <Col xs={1} style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                <Button
                  className="position-relative"
                  variant="primary"
                  onClick={handleClickGenerate}
                  disabled={props.isGenerating}
                >
                  <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                    {props.isGenerating ? "Generating..." : "Generate"}
                    <Spinner
                      size="sm"
                      style={{ marginLeft: '5px', display: props.isGenerating ? 'inline-block' : 'none' }}
                      variant="light"
                      animation="border"
                      role="status"
                    >
                      <span className="visually-hidden">Loading...</span>
                    </Spinner>
                  </span>
                </Button>
              </Col>
            </Row>
          </Card.Body>
          : null}
      </Card>
    </Col>
  );
}
export default TextPromptView;