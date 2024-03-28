import React from "react";

import { Dropdown } from "react-bootstrap";
import { ButtonGroup } from "react-bootstrap";
import { instrumentMap } from "./InstrumentList";

import { getIconName } from "./IconMapping";


const InstListDropdown = (props) => {

    const instNumToName = () => {
        if (props.addInstNum === 999) {
            return "Random";
        } else if (props.addInstNum === -1) {
            return "Drums";
        } else {
            return instrumentMap[props.addInstNum];
        }
    };

    return (
        <Dropdown
            size={props.isMobileDevice && "sm"}
            as={ButtonGroup}
            className="float-end"
            variant="outline-dark"
            drop="up"
        >
            <Dropdown.Toggle
                variant="outline-dark"
                id="dropdown-basic"
            >
                <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                    <img src={`./inst_icons/${getIconName(props.addInstNum)}.png`} width="14px" style={{ marginRight: "5px" }} />
                    {instNumToName()}
                </span>
            </Dropdown.Toggle>
            <Dropdown.Menu
                style={{
                    maxHeight: "500px",
                    overflowY: "auto",
                }}>
                <Dropdown.Item
                    as="button"
                    onClick={() => {
                        props.setAddInstNum(999);
                    }}
                >
                    <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                        <img src={`./inst_icons/${getIconName(999)}.png`} width="20px" style={{ marginRight: "5px" }} />
                        Random
                    </span>
                </Dropdown.Item>
                <Dropdown.Item
                    as="button"
                    onClick={() => {
                        props.setAddInstNum(-1);
                    }}
                    disabled={props.currentInstruments.includes(-1)}
                >
                    <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                        <img src={`./inst_icons/${getIconName(-1)}.png`} width="20px" style={{ marginRight: "5px" }} />
                        Drums
                    </span>
                </Dropdown.Item>
                {Object.entries(instrumentMap).map(([key, value]) => (
                    <Dropdown.Item
                        as="button"
                        key={key}
                        onClick={() => {
                            props.setAddInstNum(parseInt(key));
                        }}
                        disabled={props.currentInstruments.includes(parseInt(key))}
                    >
                        <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                            <img src={`./inst_icons/${getIconName(parseInt(key))}.png`} width="20px" style={{ marginRight: "5px" }} />
                            {value}
                        </span>
                    </Dropdown.Item>
                ))}
            </Dropdown.Menu>
        </Dropdown>
    )
}

export default InstListDropdown;