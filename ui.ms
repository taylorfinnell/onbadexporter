startFrame = animationRange.start.frame
endFrame = animationRange.end.frame
outputPath = ""

rollout OpenNovaRoll "OpenNova Export" width:220 height:300
(
    spinner spn_startFrame "Start Frame:" range:[-5,100000,startFrame] type:#integer pos:[10,10] width:200
    spinner spn_endFrame "End Frame:" range:[-5,100000,endFrame] type:#integer pos:[10,50] width:200
    button btn_selectPath "Select Output File..." pos:[10,90] width:200
    edittext et_outputPath "" pos:[0,0] width:0 height:0 visible:false
    checkbox chk_console "Dump to Console" pos:[10,130] width:200 checked:false
    checkbox chk_looped "Looped" pos:[10,160] width:200 checked:true
    checkbox chk_trans "Translation" pos:[10,180] width:200 checked:true
    checkbox chk_bit3 "Unk1" pos:[10,200] width:200 checked:false
    checkbox chk_bit4 "Unk2" pos:[10,220] width:200 checked:false
    button btn_export "Export" pos:[10,260] width:200

    on btn_selectPath pressed do
    (
        local fname = getSaveFileName caption:"Select Output BAD File" types:"BAD Files (*.bad)|*.bad|All Files (*.*)|*.*|"
        if fname != undefined do
        (
            et_outputPath.text = fname
            format "Selected output file: %\n" fname
        )
    )

    on btn_export pressed do
    (
        local sf = spn_startFrame.value
        local ef = spn_endFrame.value
        local path = et_outputPath.text

        if path == "" then
        (
            messageBox "No output file selected!"
            return()
        )

        python.execute "export_bad()"
    )
)
createDialog OpenNovaRoll
