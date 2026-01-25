# Data Model: Professional Landing Page for Docusaurus Site

## Landing Page Component Structure

**Entity**: LandingPage
- **Fields**:
  - title: string (book title)
  - description: string (short description of the book/course)
  - courseFocus: string (specific focus area of the textbook)
  - ctaButtons: array (list of CTA button objects)
  - heroImage: string (optional hero image path)
  - layoutConfig: object (responsive layout configuration)

**Validation Rules**:
- title must be non-empty
- description must be concise (under 200 characters)
- courseFocus must clearly describe the textbook's focus
- ctaButtons must contain at least one valid button

## CTA Button Structure

**Entity**: CTAButton
- **Fields**:
  - text: string (button text, e.g., "Read Docs", "Start Module 1", "GitHub")
  - link: string (URL or route to navigate to)
  - variant: string (button style variant)
  - target: string (link target, e.g., "_self", "_blank")

**Validation Rules**:
- text must be non-empty
- link must be a valid URL or internal route
- variant must be one of predefined styles

## Hero Section Structure

**Entity**: HeroSection
- **Fields**:
  - title: string (main heading for the hero section)
  - subtitle: string (optional subtitle)
  - description: string (detailed description text)
  - primaryButton: CTAButton (primary call-to-action)
  - secondaryButton: CTAButton (secondary call-to-action, optional)

**Relationships**:
- LandingPage contains one HeroSection
- HeroSection contains multiple CTAButton objects

## Responsive Layout Configuration

**Entity**: LayoutConfig
- **Fields**:
  - desktop: object (layout settings for desktop view)
  - tablet: object (layout settings for tablet view)
  - mobile: object (layout settings for mobile view)
  - breakpoints: object (CSS media query breakpoints)

**Validation Rules**:
- All breakpoints must follow standard responsive design practices
- Layout settings must ensure readability across all device sizes

## Styling Configuration

**Entity**: StylingConfig
- **Fields**:
  - typography: object (font families, sizes, weights)
  - colors: object (color palette matching documentation theme)
  - spacing: object (margin, padding, layout spacing)
  - components: object (button, link, and other component styles)

**Relationships**:
- StylingConfig applied to all components in LandingPage
- Must maintain consistency with existing documentation theme