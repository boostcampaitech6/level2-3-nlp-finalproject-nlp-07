import React from "react";
import { useState, useEffect } from "react"

import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Button from "react-bootstrap/Button";
import Spinner from "react-bootstrap/Spinner"

import { getIconName } from "../utils/IconMapping";


const SingleTrackView = (props) => {
    const [startTimes, setStartTimes] = useState([]);
    const [highlightOn, setHighlightOn] = useState(false);

    // Bar Component
    const BarComponent = (barIdx, notes) => {

        // 필요한 정보 : Bar Index, ticks per beat(ppq)
        // console.log(`Bar ${index} of track${props.idx}`);
        // console.log(`Ticks Per Beat: ${props.ticksPerBeat}`);

        const getBarNotes = (ticks) => {
            return (ticks >= barIdx * props.ticksPerBeat * props.beatsPerBar) &&
                (ticks < (barIdx + 1) * props.ticksPerBeat * props.beatsPerBar)
                ? true : false;
        }

        return (
            <div
                key={barIdx}
                style={handleBarStyle(barIdx)}
            >
                {notes.map((note, idx) => (
                    getBarNotes(note.ticks) ?
                        <div
                            key={idx}
                            style={props.handleNoteStyle(
                                idx,
                                barIdx,
                                note.time,
                                note.ticks,
                                note.durationTicks,
                                note.duration,
                                note.time == startTimes[startTimes.length - 1]
                                    ? note.time + note.duration
                                    : startTimes[startTimes.indexOf(note.time) + 1],
                                note.pitch,
                                highlightOn
                            )}>
                        </div> : null))
                }
            </div>
        )
    }

    useEffect(() => {
        let startTimeSet = new Set();
        props.track.notes.forEach((note) => {
            startTimeSet.add(note.time);
        })
        const startTimeArray = Array.from(startTimeSet);
        setStartTimes(startTimeArray);
    }, [props.track])

    const trackAreaStyle = {
        backgroundColor: props.color,
        height: '7vh',
        borderRadius: '10px',
        marginBottom: '7px',
        alignItems: "flex-end"
    }

    const handleClickRegenerate = () => {
        if (props.track.instrument.percussion === true) {
            props.setRegenInstNum(-1);
        } else {
            props.setRegenInstNum(props.track.instrument.number);
        }
        props.setRegenTrackIdx(props.idx);
        props.setRegenTrigger((prev) => prev + 1);
    }
    let instNum;
    if (props.track.instrument.percussion === true) {
        instNum = -1;
    } else {
        instNum = props.track.instrument.number;
    }

    const handleBarStyle = (barIdx) => {
        return {
            height: "100%",
            width: `${100 / props.totalBars}%`,
            position: "relative",
            display: "flex",
            alignItems: "flex-end",
            borderLeft: `${Math.ceil(barIdx / props.totalBars) * 2}px dotted white`,
        }
    }


    return (
        <Row key={props.idx}>
            <Col xs={1} className="d-flex align-items-center">
                <div className="me-2">
                    <img src={`./inst_icons/${getIconName(instNum)}.png`} width="25px" />
                </div>
                <div style={{ maxHeight: "7vh", overflowY: "hidden" }}>
                    {props.track.name}
                </div>
            </Col>
            <Col xs={1} className="d-flex align-items-center">
                <Button
                    className="float-end"
                    disabled={props.isGenerating || props.isAdding}
                    variant="outline-primary"
                    onClick={() => props.handleSoloButton(props.idx)}
                    active={props.soloTrack.includes(props.idx)}
                    size="sm"
                >
                    S
                </Button>
                <Button
                    className="float-end"
                    disabled={props.isGenerating || props.isAdding}
                    variant="outline-secondary"
                    onClick={() => props.handleMuteButton(props.idx)}
                    active={props.mutedTracks.includes(props.idx)}
                    size="sm"
                >
                    M
                </Button>
                <Button
                    className="float-end"
                    disabled={props.playing || props.isGenerating}
                    variant="outline-danger"
                    onClick={() => props.handleClickRemove(props.idx)}
                    size="sm"
                >
                    X
                </Button>
            </Col>
            <Col xs={9}>
                <Row
                    className="ps-1 pe-1 pt-2 pb-1 d-flex"
                    style={trackAreaStyle}>
                    {[...Array(props.totalBars)].map((_, index) => (
                        BarComponent(index, props.track.notes)
                    ))}
                </Row>
            </Col>
            <Col xs={1} className="d-flex align-items-center">
                <Button
                    size="sm"
                    variant="outline-primary"
                    onClick={handleClickRegenerate}
                    disabled={props.isGenerating || props.isAdding || props.playing || props.isExtending}
                    onMouseEnter={() => setHighlightOn(true)}
                    onMouseLeave={() => setHighlightOn(false)}
                >
                    {props.isAdding && props.regenTrackIdx == props.idx ?
                        <Spinner
                            className="m-0 p-0"
                            style={{ width: '0.8rem', height: '0.8rem', borderWidth: '2px' }}
                            variant="primary"
                            animation="border"
                            role="status"
                        >
                            <span className="visually-hidden">Loading...</span>
                        </Spinner> : "↺"}
                </Button>
            </Col>
        </Row>
    )
}

export default SingleTrackView;