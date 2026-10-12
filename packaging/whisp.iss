; Whisp's installer, for Inno Setup 6.7 or later.
;
; A per-user install: no administrator rights, nothing outside the user's profile.
; Build the app first, then the installer (or let the build script do both):
;
;     .\scripts\build.ps1 -Installer
;     ISCC.exe packaging\whisp.iss          (when dist\Whisp is already built)
;
; Output: dist\Whisp-Setup-<version>.exe. The version is read from Whisp.exe.

#define AppName "Whisp"
#define AppExeName "Whisp.exe"
#define AppPublisher "Whisp contributors"
; Windows AppUserModelID; must match APP_ID in src/whisp/__init__.py so the Start menu
; shortcut, taskbar button and notifications all belong to the same app.
#define AppUserModelId "Whisp.Dictation"
#define AppSourceDir AddBackslash(SourcePath) + "..\dist\Whisp"
#define AppExePath AppSourceDir + "\" + AppExeName

#if !FileExists(AppExePath)
  #error dist\Whisp\Whisp.exe is missing. Build the app first with scripts\build.ps1.
#endif
#define AppVersion GetStringFileInfo(AppExePath, "ProductVersion")
#if AppVersion == ""
  #error Couldn't read the version from dist\Whisp\Whisp.exe.
#endif

[Setup]
; AppId identifies Whisp to Windows for upgrades and uninstalling. Never change it.
AppId={{D579E2B2-4375-4A8F-B40E-C0D7FFD15827}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppCopyright=Copyright (c) 2026 {#AppPublisher}
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} Setup
VersionInfoProductName={#AppName}
VersionInfoCopyright=Copyright (c) 2026 {#AppPublisher}

; Per user, no administrator rights. {userpf} is %LOCALAPPDATA%\Programs.
PrivilegesRequired=lowest
UsedUserAreasWarning=no
DefaultDirName={userpf}\{#AppName}
DisableDirPage=auto
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Qt 6 needs Windows 10 1809 or later.
MinVersion=10.0.17763

; An update closes a running Whisp (through Windows Restart Manager) before replacing its files.
CloseApplications=force
RestartApplications=no

OutputDir=..\dist
OutputBaseFilename={#AppName}-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes

SetupIconFile=whisp.ico
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
WizardStyle=modern dynamic
WizardImageFile=wizard.png
WizardSmallImageFile=wizard-small.png

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[InstallDelete]
; Start every version from a clean app folder, so no file from an older version lingers.
; Only when the folder really holds Whisp, so a mistyped folder can never lose another app's files.
Type: filesandordirs; Name: "{app}\_internal"; Check: WhispInstalledHere

[Files]
Source: "{#AppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; AppUserModelID: "{#AppUserModelId}"; Comment: "Voice dictation that runs on your PC"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: dirifempty; Name: "{app}"

[Code]
const
  WM_QUERYENDSESSION = $0011;
  WM_ENDSESSION = $0016;
  ENDSESSION_CLOSEAPP = $00000001;
  SMTO_ABORTIFHUNG = $0002;
  RunKey = 'Software\Microsoft\Windows\CurrentVersion\Run';
  StartupApprovedKey = 'Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run';
  { The name Whisp gives its "launch at sign-in" entry (src/whisp/platform/win32/autostart.py). }
  StartupValueName = '{#AppName}';

type
  TProcessIds = array of Cardinal;

function WhispInstalledHere(): Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\{#AppExeName}'));
end;

function FindWindowEx(Parent, ChildAfter: HWND; ClassName, WindowName: Cardinal): HWND;
  external 'FindWindowExW@user32.dll stdcall';
function GetWindowThreadProcessId(Wnd: HWND; var ProcessId: Cardinal): Cardinal;
  external 'GetWindowThreadProcessId@user32.dll stdcall';
function GetClassName(Wnd: HWND; ClassName: String; MaxCount: Integer): Integer;
  external 'GetClassNameW@user32.dll stdcall';
function SendMessageTimeout(Wnd: HWND; Msg, WParam: Cardinal; LParam: Longint;
  Flags, Timeout: Cardinal; var MsgResult: Cardinal): Cardinal;
  external 'SendMessageTimeoutW@user32.dll stdcall';

{ Process ids of every Whisp started from this installation. Other copies of Whisp
  (another folder, a developer build) are left alone. }
function RunningWhisp(): TProcessIds;
var
  Locator, Service, Processes, Process: Variant;
  ExePath, Path: String;
  I, Total, Count: Integer;
begin
  SetArrayLength(Result, 0);
  Count := 0;
  ExePath := Lowercase(ExpandConstant('{app}\{#AppExeName}'));
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\CIMV2');
    Processes := Service.ExecQuery('SELECT ProcessId, ExecutablePath FROM Win32_Process WHERE Name = ''{#AppExeName}''');
    Total := Processes.Count;
    for I := 0 to Total - 1 do
    begin
      Process := Processes.ItemIndex(I);
      if not VarIsNull(Process.ExecutablePath) then
      begin
        Path := Process.ExecutablePath;
        if Lowercase(Path) = ExePath then
        begin
          SetArrayLength(Result, Count + 1);
          Result[Count] := Process.ProcessId;
          Count := Count + 1;
        end;
      end;
    end;
  except
    Log('Couldn''t look for a running Whisp: ' + GetExceptionMessage);
  end;
end;

function IsOneOf(Pid: Cardinal; const Pids: TProcessIds): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 0 to GetArrayLength(Pids) - 1 do
    if Pids[I] = Pid then
    begin
      Result := True;
      Exit;
    end;
end;

{ Qt's own windows have class names like "Qt6120QWindowIcon". Only they handle the
  end-session messages; Whisp may have none while it sits in the tray. }
function IsQtWindow(Wnd: HWND): Boolean;
var
  Name: String;
  Len: Integer;
begin
  Name := StringOfChar(' ', 256);
  Len := GetClassName(Wnd, Name, 256);
  if Len <= 0 then
    Result := False
  else
  begin
    SetLength(Name, Len);
    Result := (Copy(Name, 1, 2) = 'Qt') and (Pos('QWindow', Name) > 0);
  end;
end;

{ Sends Whisp what Windows sends when the user signs out. Whisp then quits the normal
  way: it gives other apps their sound back, stops its AI server and closes its files.
  Returns False if none of the processes has a window that can take the message. }
function AskToQuit(const Pids: TProcessIds): Boolean;
var
  Wnd: HWND;
  Pid, Reply: Cardinal;
begin
  Result := False;
  Wnd := FindWindowEx(0, 0, 0, 0);
  while Wnd <> 0 do
  begin
    Pid := 0;
    GetWindowThreadProcessId(Wnd, Pid);
    if IsOneOf(Pid, Pids) and IsQtWindow(Wnd) then
    begin
      Result := True;
      SendMessageTimeout(Wnd, WM_QUERYENDSESSION, 0, ENDSESSION_CLOSEAPP, SMTO_ABORTIFHUNG, 5000, Reply);
      SendMessageTimeout(Wnd, WM_ENDSESSION, 1, ENDSESSION_CLOSEAPP, SMTO_ABORTIFHUNG, 10000, Reply);
    end;
    Wnd := FindWindowEx(0, Wnd, 0, 0);
  end;
end;

function WaitForWhispToExit(TimeoutMs: Integer): Boolean;
var
  Waited: Integer;
begin
  Waited := 0;
  Result := GetArrayLength(RunningWhisp()) = 0;
  while (not Result) and (Waited < TimeoutMs) do
  begin
    Sleep(250);
    Waited := Waited + 250;
    Result := GetArrayLength(RunningWhisp()) = 0;
  end;
end;

procedure StopRunningWhisp();
var
  Pids: TProcessIds;
  I, ResultCode: Integer;
begin
  Pids := RunningWhisp();
  if GetArrayLength(Pids) = 0 then
    Exit;
  Log(Format('Whisp is running (%d process(es)); asking it to quit', [GetArrayLength(Pids)]));
  if AskToQuit(Pids) and WaitForWhispToExit(12000) then
  begin
    Log('Whisp quit');
    Exit;
  end;
  Pids := RunningWhisp();
  for I := 0 to GetArrayLength(Pids) - 1 do
  begin
    Log('Ending Whisp process ' + IntToStr(Pids[I]));
    Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /T /PID ' + IntToStr(Pids[I]), '', SW_HIDE,
      ewWaitUntilTerminated, ResultCode);
  end;
  if not WaitForWhispToExit(5000) then
    Log('Whisp is still running; some of its files may stay behind');
end;

{ Whisp adds itself to the startup apps (HKCU Run) when "Launch at startup" is on.
  Remove that entry if it starts this installation's Whisp. }
procedure RemoveStartupEntry();
var
  Command: String;
begin
  if not RegQueryStringValue(HKCU, RunKey, StartupValueName, Command) then
    Exit;
  if Pos(Lowercase(ExpandConstant('{app}\{#AppExeName}')), Lowercase(Command)) = 0 then
  begin
    Log('The Whisp startup entry starts another copy of Whisp; leaving it');
    Exit;
  end;
  RegDeleteValue(HKCU, RunKey, StartupValueName);
  RegDeleteValue(HKCU, StartupApprovedKey, StartupValueName);
  Log('Removed Whisp from the startup apps');
end;

{ Settings live in %APPDATA%\Whisp; history, models, downloaded runtimes and logs in
  %LOCALAPPDATA%\Whisp. Never removed without asking, and kept in silent uninstalls. }
procedure OfferToDeleteUserData();
var
  SettingsDir, DataDir, Left: String;
begin
  SettingsDir := ExpandConstant('{userappdata}\{#AppName}');
  DataDir := ExpandConstant('{localappdata}\{#AppName}');
  if (not DirExists(SettingsDir)) and (not DirExists(DataDir)) then
    Exit;
  if UninstallSilent() then
  begin
    Log('Silent uninstall: keeping the user''s Whisp settings and data');
    Exit;
  end;
  if SuppressibleMsgBox(
    'Do you also want to delete your Whisp data?' + #13#10#13#10 +
    'This removes your settings, dictionary, snippets and dictation history, and the speech and AI models you downloaded:' + #13#10#13#10 +
    SettingsDir + #13#10 + DataDir + #13#10#13#10 +
    'Choose No to keep them, for example if you plan to install Whisp again.',
    mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) <> IDYES then
  begin
    Log('Keeping the user''s Whisp settings and data');
    Exit;
  end;
  Left := '';
  if DirExists(SettingsDir) and not DelTree(SettingsDir, True, True, True) then
    Left := Left + SettingsDir + #13#10;
  if DirExists(DataDir) and not DelTree(DataDir, True, True, True) then
    Left := Left + DataDir + #13#10;
  if Left <> '' then
    SuppressibleMsgBox('Some of Whisp''s files couldn''t be deleted. You can delete these folders yourself:' + #13#10#13#10 + Left,
      mbInformation, MB_OK, IDOK)
  else
    Log('Deleted the user''s Whisp settings and data');
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  case CurUninstallStep of
    usUninstall:
      StopRunningWhisp();
    usPostUninstall:
      begin
        RemoveStartupEntry();
        OfferToDeleteUserData();
      end;
  end;
end;
