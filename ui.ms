-- First, clean up any existing UI elements
try (destroyDialog OpenNovaRoll) catch ()
if OpenNovaRoll != undefined do (
    try (
        DestroyDialog OpenNovaRoll
        globalVars.remove #OpenNovaRoll
    ) catch ()
)

-- The AnimRange struct definition
struct AnimRange (
    name = "Unnamed",
    startFrame = 0,
    endFrame = 0,
    isLooped = true,
    hasTranslation = true,
    hasUnk1 = false,
    hasUnk2 = false,
    
    fn toString = (
        local result = stringstream ""
        try (
            format "%" name to:result
            format ";" to:result
            format "%" startFrame to:result
            format ";" to:result
            format "%" endFrame to:result
            format ";" to:result
            format "%" (if isLooped then "true" else "false") to:result
            format ";" to:result
            format "%" (if hasTranslation then "true" else "false") to:result
            format ";" to:result
            format "%" (if hasUnk1 then "true" else "false") to:result
            format ";" to:result
            format "%" (if hasUnk2 then "true" else "false") to:result
        ) catch (
            print(getCurrentException())
        )
        local finalString = result as string
        close result
        finalString
    ),
    
    fn fromString str = (
        try (
            local values = filterString str ";"
            if values.count >= 7 do (
                name = values[1]
                startFrame = try(values[2] as integer) catch(0)
                endFrame = try(values[3] as integer) catch(100)
                isLooped = values[4] == "true"
                hasTranslation = values[5] == "true"
                hasUnk1 = values[6] == "true"
                hasUnk2 = values[7] == "true"
            )
        ) catch (
            print(getCurrentException())
        )
    )
)

rollout OpenNovaRoll "OpenNova Animation Export" width:480 (
    -- Storage for animation ranges
    local animRanges = #()
    
    -- Global Settings Section
    group "Export Directory" (
        edittext et_exportGlobal "" pos:[10,20] width:400
        button btn_browseGlobal "..." pos:[420,20] width:30 height:20
    )
    
    -- Animation Ranges Section
    label lbl_animations "Animation Ranges:" pos:[10,50] width:460
    listbox lb_animations "" height:12 pos:[10,70] width:460
    
    -- Management Buttons
    button btn_add "Add Animation" pos:[10,240] width:225
    button btn_remove "Remove Selected" pos:[245,240] width:225
    
    -- Animation Details Section
    group "Animation Details" (
        -- Name field
        label lbl_name "Name:" pos:[10,290] width:40
        edittext et_name "" pos:[60,290] width:410
        
        -- Frame range controls
        label lbl_start "Start Frame:" pos:[10,315] width:60
        spinner spn_start range:[-10000,10000,0] type:#integer pos:[80,315] width:100
        
        label lbl_end "End Frame:" pos:[300,315] width:60
        spinner spn_end range:[-10000,10000,100] type:#integer pos:[360,315] width:100
        
        -- Checkbox grid
        checkbox chk_looped "Looped" pos:[15,335] width:110 checked:true
        checkbox chk_trans "Translation" pos:[130,335] width:110 checked:true
        checkbox chk_unk1 "Unknown1" pos:[250,335] width:110 checked:false
        checkbox chk_unk2 "Unknown2" pos:[370,335] width:110 checked:false
    )
    
    -- Bottom Action Buttons
    button btn_saveConfig "Save Configuration" pos:[10,380] width:225
    button btn_loadConfig "Load Configuration" pos:[245,380] width:225
    button btn_export "Export All Animations" pos:[10,410] width:460 height:30

    -- Utility Functions
    fn updateListBox = (
        lb_animations.items = for anim in animRanges collect 
            (anim.name + " [" + (anim.startFrame as string) + "-" + (anim.endFrame as string) + "]")
    )

    fn clearDetails = (
        et_name.text = ""
        spn_start.value = 0
        spn_end.value = 100
        chk_looped.checked = true
        chk_trans.checked = true
        chk_unk1.checked = false
        chk_unk2.checked = false
    )

    fn updateDetails anim = (
        et_name.text = anim.name
        spn_start.value = anim.startFrame
        spn_end.value = anim.endFrame
        chk_looped.checked = anim.isLooped
        chk_trans.checked = anim.hasTranslation
        chk_unk1.checked = anim.hasUnk1
        chk_unk2.checked = anim.hasUnk2
    )

    fn saveToFile filepath = (
        try (
            local dir = getFilenamePath filepath
            makeDir dir
            
            local file = createFile filepath
            if file == undefined do (
                messageBox "Failed to create file!"
                return false
            )
            
            format "EXPORTDIR;%\n" (et_exportGlobal.text) to:file
            
            for anim in animRanges do (
                local animString = anim.toString()
                if animString != "" then (
                    format "%\n" animString to:file
                )
            )
            
            close file
            messageBox "Configuration saved successfully!"
            true
        ) catch (
            messageBox ("Error saving configuration: " + getCurrentException())
            false
        )
    )

    fn loadFromFile filepath = (
        try (
            if not (doesFileExist filepath) do (
                messageBox "Configuration file does not exist!"
                return false
            )
            
            local file = openFile filepath mode:"r"
            if file == undefined do (
                messageBox "Failed to open file!"
                return false
            )
            
            animRanges = #()
            
            local firstLine = readLine file
            if (matchPattern firstLine pattern:"EXPORTDIR;*") then (
                local parts = filterString firstLine ";"
                if parts.count >= 2 then (
                    local dir = parts[2]
                    if dir != "" then et_exportGlobal.text = dir
                )
            )
            
            while not eof file do (
                local line = readLine file
                if line.count > 0 do (
                    local newAnim = AnimRange()
                    newAnim.fromString line
                    append animRanges newAnim
                )
            )
            
            close file
            
            updateListBox()
            if animRanges.count > 0 do (
                lb_animations.selection = 1
                updateDetails animRanges[1]
            )
            
            messageBox "Configuration loaded successfully!"
            true
        ) catch (
            messageBox ("Error loading configuration: " + getCurrentException())
            false
        )
    )

    -- Event Handlers
    on btn_add pressed do (
        local newAnim = AnimRange()
        newAnim.name = et_name.text
        if newAnim.name == "" do newAnim.name = "Animation " + (animRanges.count + 1) as string
        newAnim.startFrame = spn_start.value
        newAnim.endFrame = spn_end.value
        newAnim.isLooped = chk_looped.checked
        newAnim.hasTranslation = chk_trans.checked
        newAnim.hasUnk1 = chk_unk1.checked
        newAnim.hasUnk2 = chk_unk2.checked
        append animRanges newAnim
        updateListBox()
        lb_animations.selection = animRanges.count
    )

    on btn_remove pressed do (
        local sel = lb_animations.selection
        if sel > 0 do (
            deleteItem animRanges sel
            updateListBox()
            if animRanges.count > 0 then (
                lb_animations.selection = (sel - 1) + 1
                updateDetails animRanges[lb_animations.selection]
            ) else (
                clearDetails()
            )
        )
    )

    on lb_animations selected sel do (
        if sel > 0 do updateDetails animRanges[sel]
    )

    on et_name changed txt do (
        local sel = lb_animations.selection
        if sel > 0 do (
            animRanges[sel].name = txt
            updateListBox()
            lb_animations.selection = sel
        )
    )

    on spn_start changed val do (
        local sel = lb_animations.selection
        if sel > 0 do (
            animRanges[sel].startFrame = val
            updateListBox()
        )
    )

    on spn_end changed val do (
        local sel = lb_animations.selection
        if sel > 0 do (
            animRanges[sel].endFrame = val
            updateListBox()
        )
    )

    on chk_looped changed state do (
        local sel = lb_animations.selection
        if sel > 0 do animRanges[sel].isLooped = state
    )

    on chk_trans changed state do (
        local sel = lb_animations.selection
        if sel > 0 do animRanges[sel].hasTranslation = state
    )

    on chk_unk1 changed state do (
        local sel = lb_animations.selection
        if sel > 0 do animRanges[sel].hasUnk1 = state
    )

    on chk_unk2 changed state do (
        local sel = lb_animations.selection
        if sel > 0 do animRanges[sel].hasUnk2 = state
    )

    on btn_browseGlobal pressed do (
        local dir = getSavePath caption:"Select Global Export Directory"
        if dir != undefined then (
            et_exportGlobal.text = dir
        )
    )

    on btn_saveConfig pressed do (
        local fname = getSaveFileName caption:"Save Configuration" types:"Config Files (*.cfg)|*.cfg|All Files (*.*)|*.*|"
        if fname != undefined do saveToFile fname
    )

    on btn_loadConfig pressed do (
        local fname = getOpenFileName caption:"Load Configuration" types:"Config Files (*.cfg)|*.cfg|All Files (*.*)|*.*|"
        if fname != undefined do loadFromFile fname
    )

    on btn_export pressed do (
        if animRanges.count == 0 then (
            messageBox "No animations to export!"
            return()
        )
        
        local globalDir = et_exportGlobal.text
        if globalDir == "" then (
            messageBox "Global export directory not set!"
            return()
        )
        
        makeDir globalDir
        
        for i = 1 to animRanges.count do (
            local anim = animRanges[i]
            local flags = 0
            if anim.isLooped do flags += 1
            if anim.hasTranslation do flags += 2
            if anim.hasUnk1 do flags += 4
            if anim.hasUnk2 do flags += 8
            
            local outputPath = globalDir + "\\" + anim.name + ".bad"
            local command = stringstream ""
			local safeOutputPath = substituteString outputPath "\\" "\\\\"

            local command = stringstream ""
			format "export_bad('%',%,%,%,'%')" anim.name anim.startFrame anim.endFrame flags safeOutputPath to:command
			local commandStr = command as string

			print(commandStr)
            close command
            
            if (python != undefined and python.execute != undefined) then (
                try (
                    python.execute commandStr
                ) catch (
					print(getCurrentExceptionStackTrace())
                    messageBox ("Error executing Python export: " + getCurrentException())
                )
            ) else (
                messageBox "Python integration is not available!"
            )
        )
    )
)

fn createOpenNovaUI = (
    try (destroyDialog OpenNovaRoll) catch()
    createDialog OpenNovaRoll style:#(#style_titlebar, #style_border, #style_sysmenu)
)

-- Initialize the UI
createOpenNovaUI()