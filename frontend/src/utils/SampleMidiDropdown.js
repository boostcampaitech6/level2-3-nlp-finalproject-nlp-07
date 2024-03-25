import React from "react";

import Dropdown from "react-bootstrap/Dropdown";
import DropdownButton from "react-bootstrap/DropdownButton";
import ButtonGroup from "react-bootstrap/ButtonGroup";


const SampleMidiDropdown = (props) => {

    return (
        <DropdownButton
            as={ButtonGroup}
            className="float-start ms-2"
            title={props.sampleTitle}
            variant="outline-dark"
            disabled={props.isGenerating || props.isAdding}
        >
            <Dropdown.Item
                as="button"
                key="0"
                onClick={() => {
                    props.handleLoadSampleMidi("./sample_midis/bodynsoul_sample.mid");
                    props.setSampleTitle("Body N Soul");
                }}
            >
                <span>Jazz - Body And Soul</span>
            </Dropdown.Item>
            <Dropdown.Item
                as="button"
                key="1"
                onClick={() => {
                    props.handleLoadSampleMidi("./sample_midis/twentieth_sample.mid");
                    props.setSampleTitle("20th Century Stomp");
                }}
            >
                <span>Jazz - 20th Century Stomp</span>
            </Dropdown.Item>
        </DropdownButton>
    )
}

export default SampleMidiDropdown;