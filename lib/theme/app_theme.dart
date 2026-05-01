import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// Design system for the Fiscaliza app.
///
/// Usage:
///   MaterialApp(
///     theme: AppTheme.light(),
///     darkTheme: AppTheme.dark(),
///   )
///
/// Fixed color constants (brightness-independent) are static on this class.
/// Brightness-dependent colors live in the ColorScheme and should be accessed
/// via `Theme.of(context).colorScheme`.
class AppTheme {
  AppTheme._(); // prevent instantiation

  // -------------------------------------------------------------------------
  // Fixed color constants — same in both themes
  // -------------------------------------------------------------------------

  static const Color emerald = Color(0xFF10B981);
  static const Color errorRed = Color(0xFFEF4444);
  static const Color activeLightBg = Color(0xFFECFDF5);
  static const Color activeDarkBg = Color(0x1F10B981);
  static const Color activeLightText = Color(0xFF065F46);
  static const Color activeDarkText = Color(0xFF10B981);

  // -------------------------------------------------------------------------
  // Static TextStyles — font/size/weight only, no color baked in.
  // Apply color at the call site: AppTheme.titleStyle.copyWith(color: ...)
  // -------------------------------------------------------------------------

  static TextStyle get titleStyle => GoogleFonts.inter(
        fontSize: 18,
        fontWeight: FontWeight.w700,
      );

  static TextStyle get valueStyle => GoogleFonts.inter(
        fontSize: 18,
        fontWeight: FontWeight.w700,
      );

  static TextStyle get labelStyle => GoogleFonts.inter(
        fontSize: 11,
        fontWeight: FontWeight.w600,
        letterSpacing: 1.0,
      );

  static TextStyle get bodyStyle => GoogleFonts.inter(
        fontSize: 13,
        fontWeight: FontWeight.w400,
      );

  static TextStyle get secondaryStyle => GoogleFonts.inter(
        fontSize: 12,
        fontWeight: FontWeight.w400,
      );

  static TextStyle get buttonStyle => GoogleFonts.inter(
        fontSize: 14,
        fontWeight: FontWeight.w600,
      );

  // -------------------------------------------------------------------------
  // ThemeData factories
  // -------------------------------------------------------------------------

  static ThemeData light() {
    const colorScheme = ColorScheme(
      brightness: Brightness.light,
      // 60% background / AppBar bg
      surface: Color(0xFFFFFFFF),
      // cards / 30% surface
      surfaceContainerLow: Color(0xFFF1F5F9),
      // text primary
      onSurface: Color(0xFF0F172A),
      // text secondary
      onSurfaceVariant: Color(0xFF64748B),
      // borders / dividers
      outline: Color(0xFFE2E8F0),
      // text muted
      outlineVariant: Color(0xFF94A3B8),
      // emerald — Material primary
      primary: Color(0xFF10B981),
      onPrimary: Color(0xFFFFFFFF),
      primaryContainer: Color(0xFFECFDF5),
      onPrimaryContainer: Color(0xFF065F46),
      secondary: Color(0xFF64748B),
      onSecondary: Color(0xFFFFFFFF),
      secondaryContainer: Color(0xFFF1F5F9),
      onSecondaryContainer: Color(0xFF0F172A),
      error: Color(0xFFEF4444),
      onError: Color(0xFFFFFFFF),
      errorContainer: Color(0xFFFFEDED),
      onErrorContainer: Color(0xFF7F1D1D),
    );

    return _buildTheme(colorScheme);
  }

  static ThemeData dark() {
    const colorScheme = ColorScheme(
      brightness: Brightness.dark,
      // Slate 950
      surface: Color(0xFF0F172A),
      // Slate 800
      surfaceContainerLow: Color(0xFF1E293B),
      // text primary
      onSurface: Color(0xFFF8FAFC),
      // text secondary
      onSurfaceVariant: Color(0xFF94A3B8),
      // borders / dividers
      outline: Color(0xFF334155),
      // text muted
      outlineVariant: Color(0xFF475569),
      // emerald — Material primary
      primary: Color(0xFF10B981),
      onPrimary: Color(0xFF022C22),
      primaryContainer: Color(0x1F10B981),
      onPrimaryContainer: Color(0xFF10B981),
      secondary: Color(0xFF94A3B8),
      onSecondary: Color(0xFF0F172A),
      secondaryContainer: Color(0xFF1E293B),
      onSecondaryContainer: Color(0xFFF8FAFC),
      error: Color(0xFFEF4444),
      onError: Color(0xFF0F172A),
      errorContainer: Color(0xFF3B0000),
      onErrorContainer: Color(0xFFFCA5A5),
    );

    return _buildTheme(colorScheme);
  }

  // -------------------------------------------------------------------------
  // Shared ThemeData builder
  // -------------------------------------------------------------------------

  static ThemeData _buildTheme(ColorScheme colorScheme) {
    final baseTextTheme = colorScheme.brightness == Brightness.light
        ? ThemeData.light().textTheme
        : ThemeData.dark().textTheme;

    return ThemeData(
      useMaterial3: true,
      colorScheme: colorScheme,
      fontFamily: GoogleFonts.inter().fontFamily,
      textTheme: GoogleFonts.interTextTheme(baseTextTheme),
      scaffoldBackgroundColor: colorScheme.surface,
      appBarTheme: AppBarTheme(
        elevation: 0,
        scrolledUnderElevation: 0,
        backgroundColor: colorScheme.surface,
        foregroundColor: colorScheme.onSurface,
        surfaceTintColor: Colors.transparent,
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        shadowColor: Colors.transparent,
        surfaceTintColor: Colors.transparent,
        color: colorScheme.surfaceContainerLow,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
        ),
      ),
      dividerTheme: DividerThemeData(
        color: colorScheme.outline,
        thickness: 1,
        space: 1,
      ),
    );
  }
}
