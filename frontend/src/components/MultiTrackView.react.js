import React from "react";
import { useState, useEffect } from "react";

import SingleTrackView from './SingleTrackView.react.js'

import Container from "react-bootstrap/Container";
import Card from "react-bootstrap/Card";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Button from "react-bootstrap/Button";
import ToggleButton from 'react-bootstrap/ToggleButton';
import ButtonGroup from "react-bootstrap/ButtonGroup";
import Spinner from "react-bootstrap/Spinner"

import Soundfont from "soundfont-player";
import * as Tone from "tone";

import { instrumentMap } from "../utils/InstrumentList";
import { trackColorsArray } from "../utils/trackColors.js";
import { notePositions } from "../utils/notePositions.js";


const progressBarStyle = {
  position: 'relative',
  backgroundColor: "#35a64a",
  borderRadius: "7px"
}

// 페이지를 로드할 때 하나의 AudioContext 만들어 시간을 추적하며 계속해서 사용
const audioContext = new AudioContext();

const MultiTrackView = (props) => {
  const [midiFile, setMidiFile] = useState();
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [beatsPerBar, setBeatsPerBar] = useState(4);
  const [ticksPerBeat, setTicksPerBeat] = useState(8);
  const [bpm, setBpm] = useState(120);
  const [msPerBeat, setMsPerBeat] = useState(0);
  const [totalMs, setTotalMs] = useState(0);
  const [synths, setSynths] = useState([]);
  const [soloTrack, setSoloTrack] = useState([]);
  const [mutedTracks, setMutedTracks] = useState([]);
  // const [instrumentObject, setInstrumentObject] = useState({});
  const [infillBarIdx, setInfillBarIdx] = useState();


  // currentTime 업데이트
  useEffect(() => {
    let intervalId;
    if (playing) {
      intervalId = setInterval(() => {
        setCurrentTime((prev) => prev + 100);
      }, 100);
    }
    return () => {
      clearInterval(intervalId);
    };
  }, [playing]);

  // midiFile prop 내려오면 멀티트랙으로 적용시키기
  useEffect(() => {
    if (props.midiFile) {

      // props.midiFile Logging
      console.log(props.midiFile);

      // 시간 정보 추출 및 계산
      const msPerBeat = (60 * 1000) / props.midiFile.header.tempos[0].bpm;
      const receivedBpm = props.midiFile.header.tempos[0].bpm;
      const ticksPerBeatFromMidi = props.midiFile.header.ppq;
      const beatsPerBarFromMidi = props.midiFile.header.timeSignatures[0].timeSignature[0];
      const barNumbersFromMidi = Math.round(props.midiFile.durationTicks / ticksPerBeatFromMidi / beatsPerBarFromMidi / 4) * 4; // Math.round to interval of 4
      const totalMsVal = msPerBeat * beatsPerBarFromMidi * barNumbersFromMidi;

      // console.log(`Ticks Per Beat: ${ticksPerBeatFromMidi}`);
      // console.log(`beatsPerBarFromMidi: ${beatsPerBarFromMidi}`);
      // console.log(`barNumbersFromMidi(log only, not used): ${barNumbersFromMidi}`);


      // MIDI File 및 시간 정보 적용
      setMidiFile(props.midiFile);
      setCurrentTime(0);
      setMsPerBeat(msPerBeat);
      setTotalMs(totalMsVal);
      setTicksPerBeat(ticksPerBeatFromMidi);
      setBeatsPerBar(beatsPerBarFromMidi);
      props.setTotalBars(barNumbersFromMidi);
      setBpm(receivedBpm);


      // instrumentObject 생성
      props.midiFile.tracks.forEach((track, idx) => {
        let inst = instrumentMap[track.instrument.number];

        // 없는 악기 및 드럼 예외 처리
        if (!inst) {
          inst = "acoustic_grand_piano"; // TODO : 없는 악기 piano로 임시 대체했는데, 모든 미디 악기 분류해서 mapping 해주기
        } else if (track.instrument.percussion === true) {
          inst = "synth_drum" // Drum 일단 대체
        }

        Soundfont.instrument(audioContext, inst).then(function (play) {
          props.setInstrumentObject((prev) => {
            return { ...prev, [idx]: play };
          });
        });
      })
    }
  }, [props.midiFile]);


  // 총 duration 시간 넘어가면 자동으로 재생 멈추게 하기
  useEffect(() => {
    if (currentTime >= totalMs) {
      setPlaying(false);
      setCurrentTime(0);
    }
  });

  // Solo 및 Mute 트랙 볼륨 처리
  useEffect(() => {
    if (props.instrumentObject) {
      // 1. Solo가 켜 있을 때
      if (soloTrack.length > 0) {
        Object.entries(props.instrumentObject).forEach(([idx, inst]) => {
          if (soloTrack.includes(Number(idx))) {
            inst.out.gain.value = 1
          } else {
            inst.out.gain.value = 0
          }
        })
      } else if (soloTrack.length == 0) {
        // 2-1. Solo가 꺼 있고, Mute는 켜 있을 때
        if (mutedTracks.length > 0) {
          Object.entries(props.instrumentObject).forEach(([idx, inst]) => {
            if (mutedTracks.includes(Number(idx))) {
              inst.out.gain.value = 0
            } else {
              inst.out.gain.value = 1
            }
          })
        } else if (mutedTracks.length == 0) {
          // 2-2. Solo도 꺼 있고, Mute도 꺼 있을 때
          Object.entries(props.instrumentObject).forEach(([idx, inst]) => {
            inst.out.gain.value = 1
          })
        }
      }
    }
  }, [props.instrumentObject, soloTrack, mutedTracks])


  // ======== Midi Playing / Editing Functions


  // Play Midi in soundfont instruments
  const playInstrument = () => {
    const acTime = audioContext.currentTime;
    console.log(`bpm: ${bpm}`)
    midiFile &&
      midiFile.tracks.forEach((track, idx) => {
        let inst = instrumentMap[track.instrument.number];
        if (!inst) {
          inst = "marimba";
        }
        const notes_arr = [];
        track.notes.forEach((note) => {
          note.time * 1000 >= currentTime &&
            notes_arr.push({
              time: note.time - currentTime / 1000,
              note: note.name,
              duration: note.duration,
            });
        });
        props.instrumentObject[idx].schedule(acTime, notes_arr);
      });
  };

  // Pause Instrument at current position
  const pauseInstrument = () => {
    Object.entries(props.instrumentObject).forEach(([idx, inst]) => {
      inst.stop()
    })
  }

  // Stop Button
  const stopInstrument = () => {
    Object.entries(props.instrumentObject).forEach(([idx, inst]) => {
      inst.stop()
    })
    setCurrentTime(0);
  };

  const playMidi = () => {
    // const synths = [];
    if (!playing && midiFile) {
      const now = Tone.now();

      midiFile.tracks.forEach((track, idx) => {
        if (soloTrack.length > 0 && !soloTrack.includes(idx)) {
          return;
        }

        if (mutedTracks.includes(idx)) {
          return;
        }

        //create a synth for each track
        const synth = new Tone.PolySynth(Tone.Synth, {
          envelope: {
            attack: 0.02,
            decay: 0.1,
            sustain: 0.2,
            release: 0.4,
          },
        }).toDestination();
        synths.push(synth);
        //schedule all of the events
        track.notes.forEach((note) => {
          note.time * 1000 >= currentTime &&
            synth.triggerAttackRelease(
              note.name,
              note.duration,
              note.time + now - currentTime / 1000, // 왜 되지...?
              note.velocity
            );
        });
      });
    } else {
      //dispose the synth and make a new one
      while (synths.length) {
        const synth = synths.shift();
        synth.disconnect();
      }
    }
  };

  const removeTrack = (trackNum) => {
    const newMidi = midiFile.clone()
    newMidi.tracks.splice(trackNum, 1);
    props.setMidiFile(newMidi);
  };

  const assignTrackColor = (idx) => {
    return trackColorsArray[idx % trackColorsArray.length]
  }

  // Sub Components
  const BarHeaderComponent = (index) => {

    const headerStyle = {
      width: `${100 / props.totalBars}%`,
      color: "white",
      fontSize: "0.7rem",
      opacity: 0.7,
      backgroundColor: index === infillBarIdx ? "#7591ff" : "transparent",
      cursor: index === infillBarIdx ? "pointer" : "auto"
    }

    return (
      <div
        key={index}
        style={headerStyle}
        onMouseEnter={() => { setInfillBarIdx(index) }}
        onMouseLeave={() => { setInfillBarIdx(null) }}
        onClick={() => { props.handleClickInfill(index) }}
      >
        {
          props.infillHighlightBar === index && props.isInfilling ?
            <div>
              <span>{index + 1}</span>
              <span style={{ color: "white", fontWeight: "bold", textAlign: "center", display: "inline-block", width: "90%", textAlign: "center" }}>
                <Spinner
                  // size="sm"
                  className="m-0 p-0"
                  style={{ width: '0.8rem', height: '0.8rem', borderWidth: '2px', marginLeft: '5px', display: props.isInfilling ? 'inline-block' : 'none' }}
                  variant="light"
                  animation="border"
                  role="status"
                >
                  <span className="visually-hidden">Loading...</span>
                </Spinner>
              </span>
            </div>
            :
            infillBarIdx === index ?
              <div>
                <span style={{ width: "10%" }}>
                  {index + 1}
                </span>
                <span style={{ color: "white", fontWeight: "bold", textAlign: "center", display: "inline-block", width: "90%", textAlign: "center" }}>
                  ↺
                </span>
              </div>
              :
              <span>{index + 1}</span>
        }
      </div>
    )
  }


  // Event Handlers
  const handleClickPlay = () => {
    setPlaying((prev) => !prev);
    playMidi();
  };
  const handleClickRewind = () => {
    currentTime - msPerBeat > 0 && setCurrentTime((prev) => prev - msPerBeat);
  };

  const handleClickForward = () => {
    currentTime + msPerBeat < totalMs &&
      setCurrentTime((prev) => prev + msPerBeat);
  };

  const handleClickBeginning = () => {
    setCurrentTime(0);
  };

  const handleClickEnd = () => {
    setCurrentTime(totalMs);
  };

  const handleClickRemove = (trackNum) => {
    if (window.confirm(`Are you sure to delete the track?`)) {
      removeTrack(trackNum);
    } else {
      return;
    }
  };

  const handleClickPlayInstrument = () => {
    if (playing) {
      pauseInstrument();
    } else {
      playInstrument();
    }
    setPlaying((prev) => !prev);
  };

  const handleClickStopInstrument = () => {
    if (playing) {
      setPlaying((prev) => !prev);
      stopInstrument();
    } else {
      setCurrentTime(0);
    }
  };

  const handleNoteStyle = (noteIdx, barIdx, time, startTicks, durationTicks, duration, nextStartTime, pitch, highlightOn) => {
    const currentTimeSec = currentTime / 1000;
    const currentBar = Math.floor(currentTime / (totalMs / props.totalBars));
    const ticksPerBar = ticksPerBeat * beatsPerBar;
    const totalTicks = ticksPerBar * props.totalBars;
    const durationPercent = (durationTicks / (ticksPerBar)) * 100; // Tick based
    const leftPercent = (startTicks - (ticksPerBar * barIdx)) / ticksPerBar * 100; // Tick based
    const noteHeight = 14;


    let borderStyle;
    let divColor;
    let widthPercent;
    let boxShadow = "none";

    // Note 스타일 처리
    if (time < currentTimeSec
      && currentTimeSec <= nextStartTime
      && Math.floor(startTicks / ticksPerBar) <= currentBar
      && Math.floor(startTicks + durationTicks / ticksPerBar) >= currentBar
    ) { // 연주중인 음표 빨간색 표시
      divColor = "#ffbaba";
      borderStyle = `1px solid #eb4b5d`;
    } else if (highlightOn && barIdx >= props.barsToRegen[0] && barIdx <= props.barsToRegen[1]) { // Regenerate 영역 파란색 하이라이트 표시
      divColor = "#e3e5fc";
      borderStyle = `1px solid #a4a7fc`;
    } else if (props.isInfilling && barIdx == props.infillHighlightBar) { // Bar Infill 진행중인 노트 회색 음영 처리
      divColor = "#d9d9d9";
      borderStyle = `1px solid #949494`;
    } else if (barIdx === infillBarIdx) { // Bar Infill 영역 파란색 하이라이트 표시
      divColor = "#e3e5fc";
      borderStyle = `1px solid #a4a7fc`;
    } else { // 기본 음표 하얀색으로 표시
      divColor = "white";
      borderStyle = `1px solid #adadad`;
    }

    // 마지막 음 duration이 경계 넘어가는 경우 예외 처리
    // if (leftPercent + durationPercent > 100) {
    if (startTicks + durationTicks > totalTicks) {
      // widthPercent = `${100 - leftPercent}%`;
      widthPercent = `${(totalTicks - startTicks) / ticksPerBar * 100}%`;
    } else {
      widthPercent = `${durationPercent}%`;
    }

    return {
      position: "absolute",
      float: "left",
      height: `${noteHeight}%`,
      border: borderStyle,
      backgroundColor: divColor,
      width: widthPercent,
      left: `${leftPercent}%`,
      bottom: `${notePositions[pitch] / 100 * (100 - noteHeight)}%`,
      boxShadow: boxShadow
    };
  }

  const handleProgressBar = () => {
    return {
      position: 'absolute',
      height: '100%',
      top: 0,
      color: "white",
      marginLeft: `${(currentTime / totalMs) * 100}%`,
      display: 'flex',
      alignItems: 'center', // Vertically center the content
    };
  };

  const handleSoloButton = (idx) => {
    if (soloTrack.includes(idx)) {
      const newSoloTrack = [...soloTrack].filter((track) => track !== idx);
      setSoloTrack(newSoloTrack);
    } else if (mutedTracks.includes(idx)) {
      const newMutedTrack = [...mutedTracks].filter((track) => track !== idx);
      setMutedTracks(newMutedTrack);
      const newSoloTrack = [...soloTrack].concat(idx);
      setSoloTrack(newSoloTrack);
    } else {
      const newSoloTrack = [...soloTrack].concat(idx);
      setSoloTrack(newSoloTrack);
    }
  };

  const handleMuteButton = (idx) => {
    if (mutedTracks.includes(idx)) {
      const newMutedTrack = [...mutedTracks].filter((track) => track !== idx);
      setMutedTracks(newMutedTrack);
    } else if (soloTrack.includes(idx)) {
      const newSoloTrack = [...soloTrack].filter((track) => track !== idx);
      setSoloTrack(newSoloTrack);
      const newMutedTrack = [...mutedTracks].concat(idx);
      setMutedTracks(newMutedTrack);
    } else {
      const newMutedTrack = [...mutedTracks].concat(idx);
      setMutedTracks(newMutedTrack);
    }
  };

  return (
    <>
      <Row>
        <Col className="mt-0">
          <Button
            variant="dark"
            onClick={handleClickPlayInstrument}
            disabled={props.isGenerating}
          >
            {playing ? "PAUSE" : "PLAY"}
          </Button>
          <Button
            className="ms-2"
            variant="dark"
            onClick={handleClickStopInstrument}
            disabled={props.isGenerating}
          >
            ■
          </Button>
          <Button
            className="ms-2"
            variant="dark"
            onClick={handleClickBeginning}
            disabled={props.isGenerating}
          >
            ◀◀
          </Button>
          <Button
            className="ms-2"
            variant="dark"
            onClick={handleClickRewind}
            disabled={props.isGenerating}
          >
            ◀
          </Button>
          <Button
            className="ms-2"
            variant="dark"
            onClick={handleClickForward}
            disabled={props.isGenerating}
          >
            ▶
          </Button>
          {/* <Button
            className="ms-2"
            variant="dark"
            onClick={handleClickEnd}
            disabled={props.isGenerating}
          >
            ▶▶
          </Button> */}
          <Button
            disabled={props.isGenerating}
            className="ms-2 float-middle"
            variant="dark"
            onClick={handleClickPlay}
          >
            {playing ? "PAUSE" : "PLAY 8bit"}
          </Button>
          <ButtonGroup
            className="float-end"
            hidden={props.totalBars === 4}
          >
            <ToggleButton
              key="front"
              size="sm"
              type="radio"
              variant="outline-primary"
              name="radio"
              value={1}
              checked={JSON.stringify(props.barsToRegen) === JSON.stringify([0, 3])}
              onClick={() => props.setBarsToRegen([0, 3])}
            >
              1-4
            </ToggleButton>
            <ToggleButton
              key="back"
              size="sm"
              type="radio"
              variant="outline-primary"
              name="radio"
              value={2}
              checked={JSON.stringify(props.barsToRegen) === JSON.stringify([4, 7])}
              onClick={() => props.setBarsToRegen([4, 7])}
            >
              5-8
            </ToggleButton>
          </ButtonGroup>
        </Col>
      </Row>
      <Row className="mt-3" style={{ color: "gray" }}>
        <Col xs={2}>
          {/* <div>Total Time: {(totalMs / 1000).toFixed(1)} (s)</div> */}
          <div>
            <span>{(currentTime / 1000).toFixed(1)} (s)</span>
            <span> / {(totalMs / 1000).toFixed(1)} (s), </span>
            <span>BPM: {Math.round(bpm)}</span>
          </div>
          <div>

          </div>
        </Col>
        <Col xs={9} style={progressBarStyle} className="mb-2 p-0">
          <div style={{ display: 'flex', paddingLeft: "5px" }}>
            {midiFile && [...Array(props.totalBars)].map((_, index) => (
              BarHeaderComponent(index)
            ))}
          </div>
          <div style={handleProgressBar()}>▼</div>
        </Col>
        <Col xs={1}>
        </Col>
      </Row>
      {midiFile
        ? midiFile.tracks.map((track, idx) =>
          track.notes.length > 0 ? ( // note가 있는 트랙만 표시
            <SingleTrackView
              key={idx}
              idx={idx}
              track={track}
              playing={playing}
              totalMs={totalMs}
              bpm={bpm}
              ticksPerBeat={ticksPerBeat}
              beatsPerBar={beatsPerBar}
              totalBars={props.totalBars}
              soloTrack={soloTrack}
              mutedTracks={mutedTracks}
              instrumentTrack={props.instrumentObject[idx]}
              regenTrackIdx={props.regenTrackIdx}
              isGenerating={props.isGenerating}
              isAdding={props.isAdding}
              isExtending={props.isExtending}
              color={assignTrackColor(idx)}
              handleClickRemove={handleClickRemove}
              handleSoloButton={handleSoloButton}
              handleMuteButton={handleMuteButton}
              handleNoteStyle={handleNoteStyle}
              setRegenTrackIdx={props.setRegenTrackIdx}
              setRegenInstNum={props.setRegenInstNum}
              setRegenTrigger={props.setRegenTrigger}
            />
          ) : null
        )
        :
        <Container>
          <Card className="mt-3" style={{ border: "none" }}>
            <Card.Body className="text-center">
              <img src="./inst_icons/conductor.png" width="80px" style={{ opacity: 0.8 }} />
              <h4 className="mt-3" style={{ color: "#3b3b3b" }}>Let's start making some music!</h4>
            </Card.Body>
          </Card>
        </Container>
      }
    </>
  );
};

export default MultiTrackView;