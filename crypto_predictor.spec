# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(['crypto_predictor.py'],
             pathex=['C:\\Users\\abdul\\OneDrive\\Desktop\\ABDULLAH\\8th sem\\FYP'],
             binaries=[],
             datas=[],
             hiddenimports=['matplotlib.backends.backend_tkagg'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='crypto_predictor',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          upx_include=[],
          runtime_tmpdir=None,
          console=True)
