export const getIconName = (instNum) => {
    switch (true) {
        case (instNum === -1):
            return "drums";
        case (instNum >= 0 && instNum <= 7):
            return "piano";
        case (instNum >= 8 && instNum <= 15):
            return "chromatic_perc";
        case (instNum >= 16 && instNum <= 23):
            return "organ";
        case (instNum >= 24 && instNum <= 31):
            return "guitar";
        case (instNum >= 32 && instNum <= 39):
            return "bass";
        case (instNum >= 40 && instNum <= 49):
            return "string";
        case (instNum >= 50 && instNum <= 51):
            return "synth";
        case (instNum >= 52 && instNum <= 54):
            return "choir";
        case (instNum === 55): // Orchestra Hit
            return "synth"; 
        case (instNum >= 56 && instNum <= 63):
            return "brass"; 
        case (instNum >= 64 && instNum <= 71):
            return "reed"; 
        case (instNum >= 72 && instNum <= 79):
            return "pipe"; 
        case (instNum >= 80 && instNum <= 87): // Synth Lead
            return "synth"; 
        case (instNum >= 88 && instNum <= 95): // Synth Pad
            return "synth"; 
        case (instNum >= 96 && instNum <= 103): // Synth Effects
            return "mixer"; 
        case (instNum >= 104 && instNum <= 111): // World
            return "sitar"; 
        case (instNum >= 112 && instNum <= 119): // Percussions
            return "congas"; 
        case (instNum >= 120 && instNum <= 127): // Percussions
            return "soundeffect"; 
        case (instNum === 999): // Random
            return "random"; 
        default:
            return "piano";
    }
}